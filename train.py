import os
import time
# import warnings
# warnings.filterwarnings("ignore")
from functools import partial
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# from dataset_sc import load_Speech_commands
# from dataset_ljspeech import load_LJSpeech
from dataloaders import dataloader
from utils import find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory

from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor
from generate import generate

from models import construct_model

import torchaudio

FREQ = 8000

def distributed_train(rank, num_gpus, group_name, cfg):
    # Initialize logger
    if rank == 0 and cfg.wandb is not None:
        wandb_cfg = cfg.pop("wandb")
        wandb.init(
            **wandb_cfg, config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Distributed running initialization
    dist_cfg = cfg.pop("distributed")
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_cfg)

    train(
        rank=rank, num_gpus=num_gpus,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        generate_cfg=cfg.generate,
        **cfg.train,
    )

def train(
    rank, num_gpus,
    diffusion_cfg, model_cfg, dataset_cfg, generate_cfg, # dist_cfg, wandb_cfg, # train_cfg,
    ckpt_iter, n_iters, iters_per_ckpt, iters_per_logging,
    learning_rate, batch_size_per_gpu, diffuse = True,
    cold = True,
    # n_samples,
    name=None,
    # mel_path=None,
):
    """
    Parameters:
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automitically selects the maximum iteration if 'max' is selected
    n_iters (int):                  number of iterations to train, default is 1M
    iters_per_ckpt (int):           number of iterations to save checkpoint,
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate
    batch_size_per_gpu (int):       batchsize per gpu, default is 2 so total batchsize is 16 with 8 gpus
    n_samples (int):                audio samples to generate and log per checkpoint
    name (str):                     prefix in front of experiment name
    mel_path (str):                 for vocoding, path to mel spectrograms (TODO generate these on the fly)
    """

    local_path, checkpoint_directory = local_directory(name, model_cfg, diffusion_cfg, dataset_cfg, 'checkpoint')

    # map diffusion hyperparameters to gpu
    diffusion_hyperparams   = calc_diffusion_hyperparams(**diffusion_cfg, fast=False)  # dictionary of all diffusion hyperparameters

    # load training data
    trainloader = dataloader(dataset_cfg, batch_size=batch_size_per_gpu, num_gpus=num_gpus, unconditional=model_cfg.unconditional)
    valloader = dataloader(dataset_cfg, batch_size=batch_size_per_gpu, num_gpus=num_gpus, unconditional=model_cfg.unconditional)
    valloader = iter(valloader)
    print('Data loaded')

    # predefine model
    net = construct_model(model_cfg).cuda()
    print_size(net, verbose=False)

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(checkpoint_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(checkpoint_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # HACK to reset learning rate
                optimizer.param_groups[0]['lr'] = learning_rate

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            print(f"Model checkpoint found at iteration {ckpt_iter}, but was not successfully loaded - training from scratch.")
            ckpt_iter = -1
    else:
        print('No valid checkpoint model found - training from scratch.')
        ckpt_iter = -1

    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        epoch_loss = 0.
        for data in tqdm(trainloader, desc=f'Epoch {n_iter // len(trainloader)}'):
            if model_cfg["unconditional"]:
                audio, syn_audio, proll, _ = data
                

                #set audio as random noise
                #audio = torch.randn(audio1.shape)
                syn_audio = syn_audio.cuda()
                # load audio
                audio = audio.cuda()

                # normalize audio
                #audio = audio / torch.max(torch.abs(audio))

                assert not torch.isnan(audio).any(), "Audio is NaN"
                mel_spectrogram = None
            else:
                mel_spectrogram, audio = data
                mel_spectrogram = mel_spectrogram.cuda()
                audio = audio.cuda()

            # back-propagation
            optimizer.zero_grad()
            if diffuse and not cold:
                loss = training_loss(net, nn.MSELoss(), audio, syn_audio, diffusion_hyperparams, mel_spec=mel_spectrogram)
            elif diffuse and cold:
                loss = training_loss_cold(net, nn.MSELoss(), audio, syn_audio, diffusion_hyperparams, mel_spec=mel_spectrogram)
            else:
                loss = training_loss_noDiffusion(net, nn.MSELoss(), audio, syn_audio, diffusion_hyperparams, mel_spec=mel_spectrogram)
            
            assert not torch.isnan(loss).any(), "Loss is NaN"

            if torch.isnan(loss).any():
                print("Loss is NaN")
                continue

            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            loss.backward()
            optimizer.step()

            epoch_loss += reduced_loss

            # output to log
            if n_iter % iters_per_logging == 0 and rank == 0:
                # save training loss to tensorboard
                # print("iteration: {} \treduced loss: {} \tloss: {}".format(n_iter, reduced_loss, loss.item()))
                # tb.add_scalar("Log-Train-Loss", torch.log(loss).item(), n_iter)
                # tb.add_scalar("Log-Train-Reduced-Loss", np.log(reduced_loss), n_iter)
                wandb.log({
                    'train/loss': reduced_loss,
                    'train/log_loss': np.log(reduced_loss),
                }, step=n_iter)

            # save checkpoint
            if n_iter % iters_per_ckpt == 0 and rank == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(checkpoint_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

                # Generate samples
                # if model_cfg["unconditional"]:
                #     mel_path = None
                #     mel_name = None
                # else:
                #     assert mel_path is not None
                #     mel_name=generate_cfg.mel_name # "LJ001-0001"
                if not model_cfg["unconditional"]: assert generate_cfg.mel_name is not None
                generate_cfg["ckpt_iter"] = n_iter
                valdata = next(valloader)
                audio_val, syn_val, _, _ = valdata
                #proll_val = proll_val.cuda()
                syn_val = syn_val.cuda()
                #audio_val = audio_val
                samples = generate(
                    rank, # n_iter,
                    diffusion_cfg, model_cfg, dataset_cfg,
                    name=name,
                    syn_audio=syn_val,
                    diffuse = diffuse,
                    cold = cold,
                    **generate_cfg,
                    # n_samples, n_iter, name,
                    # mel_path=mel_path,
                    # mel_name=mel_name,
                )
                samples = [wandb.Audio(sample.squeeze().cpu(), sample_rate=dataset_cfg['sampling_rate']) for sample in samples]
                wandb.log(
                    {'inference/audio': samples},
                    step=n_iter,
                    # commit=False,
                )

                # log audio_val and prol_val
                audio_vals = [wandb.Audio(av.squeeze().cpu(), sample_rate=dataset_cfg['sampling_rate']) for av in audio_val]
                syn_vals = [wandb.Audio(pv.squeeze().cpu(), sample_rate=dataset_cfg['sampling_rate']) for pv in syn_val]
                wandb.log(
                    {'inference/audio_val': audio_vals, 'inference/syn_val': syn_vals},
                    step=n_iter,
                    # commit=False,
                )

            n_iter += 1
        if rank == 0:
            epoch_loss /= len(trainloader)
            wandb.log({'train/loss_epoch': epoch_loss, 'train/log_loss_epoch': np.log(epoch_loss)}, step=n_iter)

    # Close logger
    if rank == 0:
        # tb.close()
        wandb.finish()

def training_loss(net, loss_fn, audio, syn_audio, diffusion_hyperparams, mel_spec=None):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    # audio = X
    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
    z = torch.normal(0, 1, size=audio.shape).cuda()
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    epsilon_theta, r = net((transformed_X, diffusion_steps.view(B,1),), mel_spec=mel_spec, syn_audio=syn_audio)  # predict \epsilon according to \epsilon_\theta

    assert not torch.isnan(epsilon_theta).any()
    return loss_fn(epsilon_theta, z)

def cold_distort_single(audio,t, T):
    """
    Distorting is frequency resampling
    """
    #new_freq = int((T-t)/T*FREQ)
    # exp decay
    t = t.clone().detach().cpu().numpy()
    # linear decay from 100% to 10%

    new_freq = int(FREQ*np.exp(-t*2.7/T))
    # resample but maintain the same length
    with torch.no_grad():
        distorted_audio = torchaudio.transforms.Resample(FREQ, new_freq)(audio.cpu())
        distorted_audio = torchaudio.transforms.Resample(new_freq, FREQ)(distorted_audio)
        # normalize
        distorted_audio = distorted_audio / torch.abs(distorted_audio).max()
    return distorted_audio.cuda()

def cold_distort(audio, t, T):
    """
    for batch of audio
    t is a tensor of shape (batchsize, 1)
    """
    distorted_audio = []
    for i in range(audio.shape[0]):
        distorted_audio.append(cold_distort_single(audio[i], t[i][0], T))
    return torch.stack(distorted_audio).cuda()


def training_loss_cold(net, loss_fn, audio, syn_audio, diffusion_hyperparams, mel_spec=None):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
    
    # audio = X
    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    # randomly sample diffusion steps from 1~T
    #t = np.random.randint(1, T, size = (B, 1))


    diffusion_steps = torch.randint(1, T, size=(B,1)).cuda()
    #print(diffusion_steps)
    t = diffusion_steps

    transformed_X = cold_distort(audio, t, T)  # compute x_t from q(x_t|x_0)
    #target_X = cold_distort(audio, t-1, T)  # compute x_t from q(x_t|x_0)

    audio_theta = net((transformed_X, diffusion_steps.view(B,1),), mel_spec=mel_spec)  # predict \epsilon according to \epsilon_\theta


    assert not torch.isnan(audio_theta).any()
    loss = loss_fn(audio_theta, audio)

    return loss



def training_loss_noDiffusion(net, loss_fn, audio_y, audio_x, diffusion_hyperparams, mel_spec=None):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]


    # audio = X
    B, C, L = audio_y.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
    #z = torch.normal(0, 1, size=audio_y.shape).cuda()
    #transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio_y + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    audio_y_pred, _ = net((audio_x, diffusion_steps.view(B,1),), mel_spec=mel_spec)  # predict \epsilon according to \epsilon_\theta
    #print("epsilon_theta.shape: ", epsilon_theta.shape)
    #if r is not None: print("r.shape: ", r.shape)
    #assert not torch.isnan(epsilon_theta).any()
    return loss_fn(audio_y_pred, audio_y)



@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    if not os.path.isdir("exp/"):
        os.makedirs("exp/")
        os.chmod("exp/", 0o775)

    num_gpus = torch.cuda.device_count()
    train_fn = partial(
        distributed_train,
        num_gpus=num_gpus,
        group_name=time.strftime("%Y%m%d-%H%M%S"),
        cfg=cfg,
    )

    if num_gpus <= 1:
        train_fn(0)
    else:
        mp.set_start_method("spawn")
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=train_fn, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

if __name__ == "__main__":
    main()
