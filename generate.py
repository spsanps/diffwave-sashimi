import os
# import json
# import sys
import time
# import subprocess
# import warnings
# warnings.filterwarnings("ignore")

from functools import partial
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import hydra
# import wandb
from omegaconf import DictConfig, OmegaConf
# from torch.utils.tensorboard import SummaryWriter # If tensorboard is preferred over wandb

from scipy.io.wavfile import write as wavwrite
# from scipy.io.wavfile import read as wavread

from model import construct_model
from util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams, local_directory, smooth_ckpt

@torch.no_grad()
def generate(
        rank,
        n_samples, # Samples per GPU
        ckpt_iter,
        name,
        diffusion_config,
        model_config,
        dataset_config,
        batch_size=None,
        ckpt_smooth=-1,
        mel_path=None, mel_name="LJ001-0001",
    ):
    """
    Generate audio based on ground truth mel spectrogram

    Parameters:
    output_directory (str):         checkpoint path
    n_samples (int):              number of samples to generate, default is 4
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automatically selects the maximum iteration if 'max' is selected
    """

    if rank is not None:
        print(f"rank {rank} {torch.cuda.device_count()} GPUs")
        torch.cuda.set_device(rank % torch.cuda.device_count())

    local_path, output_directory = local_directory(name, model_config, diffusion_config, dataset_config, 'waveforms')

    # map diffusion hyperparameters to gpu
    diffusion_hyperparams   = calc_diffusion_hyperparams(**diffusion_config, fast=True)  # dictionary of all diffusion hyperparameters

    # predefine model
    net = construct_model(model_config).cuda()
    print_size(net)
    net.eval()

    # load checkpoint
    print('ckpt_iter', ckpt_iter)
    ckpt_path = os.path.join('exp', local_path, 'checkpoint')
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    ckpt_iter = int(ckpt_iter)

    if ckpt_smooth < 0: # TODO not a good default, should be None
        try:
            model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')
            net.load_state_dict(checkpoint['model_state_dict'])
            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            raise Exception('No valid model found')
    else:
        state_dict = smooth_ckpt(ckpt_path, ckpt_smooth, ckpt_iter, alpha=None)
        net.load_state_dict(state_dict)

    # Add checkpoint number to output directory
    output_directory = os.path.join(output_directory, str(ckpt_iter))
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("saving to output directory", output_directory)

    # if rank is not None:
    #     output_directory = os.path.join(output_directory, str(rank))
    #     if not os.path.isdir(output_directory):
    #         os.makedirs(output_directory)
    #         os.chmod(output_directory, 0o775)

    if batch_size is None:
        batch_size = n_samples
    assert n_samples % batch_size == 0

    if mel_path is not None and mel_name is not None:
        # use ground truth mel spec
        try:
            ground_truth_mel_name = os.path.join(mel_path, '{}.wav.pt'.format(mel_name))
            ground_truth_mel_spectrogram = torch.load(ground_truth_mel_name).unsqueeze(0).cuda()
        except:
            raise Exception('No ground truth mel spectrogram found')
        audio_length = ground_truth_mel_spectrogram.shape[-1] * dataset_config["hop_length"]
    else:
        # predefine audio shape
        audio_length = dataset_config["segment_length"]  # 16000
        ground_truth_mel_spectrogram = None
    print(f'begin generating audio of length {audio_length} | {n_samples} samples with batch size {batch_size}')

    # inference
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    generated_audio = []

    for _ in range(n_samples // batch_size):
        _audio = sampling(
            net,
            (batch_size,1,audio_length),
            diffusion_hyperparams,
            condition=ground_truth_mel_spectrogram,
        )
        generated_audio.append(_audio)
    generated_audio = torch.cat(generated_audio, dim=0)

    end.record()
    torch.cuda.synchronize()
    print('generated {} samples shape {} at iteration {} in {} seconds'.format(n_samples,
        generated_audio.shape,
        ckpt_iter,
        int(start.elapsed_time(end)/1000)))

    # save audio to .wav
    for i in range(n_samples):
        outfile = '{}k_{}.wav'.format(ckpt_iter // 1000, n_samples*rank + i)
        wavwrite(os.path.join(output_directory, outfile),
                    dataset_config["sampling_rate"],
                    generated_audio[i].squeeze().cpu().numpy())

        # save audio to tensorboard
        # tb = SummaryWriter(os.path.join('exp', local_path, tensorboard_directory))
        # tb.add_audio(tag=outfile, snd_tensor=generated_audio[i], sample_rate=dataset_config["sampling_rate"])
        # tb.close()

    print('saved generated samples at iteration %s' % ckpt_iter)
    return generated_audio


@hydra.main(version_base=None, config_path="", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    num_gpus = torch.cuda.device_count()
    generate_fn = partial(
        generate,
        diffusion_config=cfg.diffusion_config,
        model_config=cfg.model_config,
        dataset_config=cfg.dataset_config,
        **cfg.generate_config,
    )

    if num_gpus <= 1:
        generate_fn(0)
    else:
        mp.set_start_method("spawn")
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=generate_fn, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()