## Modified based on https://github.com/pytorch/audio/blob/master/torchaudio/datasets/speechcommands.py

import os
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, models, transforms

from torch.utils.data.distributed import DistributedSampler
from scipy.io.wavfile import read as wavread

from typing import Tuple

import torchaudio
from torch.utils.data import Dataset
from torch import Tensor
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)

import glob


FREQ = 16000
SAMPLE_RATE = FREQ
MEL_FREQ = 32
LEN = 16

def files_to_list(data_path, ends_with = '.mp3'):
    """
    Load all .wav files in data_path
    """
    # Go through all subdirectories and find the files
    ret = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(ends_with):
                #files.append(file)
                ret.append(os.path.join(root, file))
    return ret, None

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    audio, sample_rate = torchaudio.load(full_path, format='mp3')

    # downsample to 16kHz
    if sample_rate != FREQ:
        audio = torchaudio.transforms.Resample(sample_rate, FREQ)(audio)

    audio = fix_length(audio, FREQ*LEN)

    return audio, sample_rate, "violin" # add label

def fix_length(tensor, length):
    assert len(tensor.shape) == 2
    channels = tensor.shape[0]
    if channels > 1:
        tensor = tensor.mean(dim=0, keepdim=True)
    if tensor.shape[1] > length:
        audio_len = tensor.shape[1]
        start = np.random.randint(0, audio_len - length)
        return tensor[:,start:start+length]
    elif tensor.shape[1] < length:
        return torch.cat([tensor, torch.zeros(1, length-tensor.shape[1])], dim=1)
    else:
        return tensor

class BachViolin(Dataset):
    """
    Create a Dataset for Bach Violin. Each item is a tuple of the form:
    waveform, sample_rate
    """

    def __init__(self, path: str):
        self.audio_paths, self.audio_files = files_to_list(path) 

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        n = n % len(self.audio_paths)
        filename = self.audio_paths[n]
        audio =  load_wav_to_torch(filename)
        assert not torch.isnan(audio[0]).any(), "NaN in audio Loader"
        return audio

    def __len__(self) -> int:
        # randomly sample parts of the audio file
        # to cover all parts of the audio file
        return len(self.audio_paths)*1000 


class BachViolinRoll(Dataset):
    """ Create a Dataset for Bach Violin. Each item is a tuple of the form:
    waveform, proll waveform, sample_rate"""

    def __init__(self, path: str):
        audio_path = os.path.join(path, "audio")
        proll_path = os.path.join(path, "proll")
        syn_path = os.path.join(path, "synthesized")

        # find all wav files in these paths using os
    
        self.audio_paths = []
        # for filename in glob.glob(os.path.join(path, '*.wav')):

        for filename in glob.glob(os.path.join(audio_path, '*.wav')):
            self.audio_paths.append(filename)

        # find corresponding synthesized wav files 
        # so that order is the same
        self.syn_paths = []
        for i in range(len(self.audio_paths)):
            self.syn_paths.append(self.audio_paths[i].replace("audio", "synthesized"))
        
        self.proll_paths = []
        for i in range(len(self.audio_paths)):
            self.proll_paths.append(self.audio_paths[i].replace("audio", "proll"))
            self.proll_paths[i] = self.proll_paths[i].replace(".wav", ".npy")
        #print(len(self.audio_paths))
        #assert False
        assert len(self.audio_paths) > 0

        #print(len(self.audio_paths))

        assert len(self.audio_paths) == len(self.syn_paths)
        assert len(self.audio_paths) == len(self.proll_paths)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        n = n % len(self.audio_paths)
        audiofile = self.audio_paths[n]
        synfile = self.syn_paths[n]

        # load audio wave
        audio, sample_rate = torchaudio.load(audiofile)
        
        # load syn wave
        syn, sample_rate2 = torchaudio.load(synfile)

        # load proll
        prollfile = self.proll_paths[n]
        proll = np.load(prollfile)



        assert sample_rate == sample_rate2
        assert sample_rate == FREQ
        assert audio.shape[1] == syn.shape[1]

        # audio is 16Khz, proll is 64Hz
        #print(audio.shape, proll.shape)
        #print(audio.shape[1], proll.shape[1])
        assert proll.shape[1]*FREQ/MEL_FREQ == audio.shape[1], f"{proll.shape[1]*FREQ/MEL_FREQ} != {audio.shape[1]}"

        # find a random LEN second segment
        audio_len = audio.shape[1]
        samp_len = FREQ*LEN
        start = np.random.randint(0, audio_len - samp_len)
        audio = audio[:,start:start+samp_len]
        syn = syn[:,start:start+samp_len]
        proll_start = int(start*MEL_FREQ//FREQ)
        proll = proll[:,proll_start:proll_start+MEL_FREQ]

        # convert to tensor
        # convert to float tensor (NOT double)
        proll = torch.from_numpy(proll).float()

        # TODO proll in tensor

        return audio, syn, proll, sample_rate

    def __len__(self) -> int:
        # randomly sample parts of the audio file
        # to cover all parts of the audio file
        return len(self.audio_paths)*1000