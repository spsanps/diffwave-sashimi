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

MAX_WAV_VALUE = 32768.0

SAMPLE_RATE = 16000

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
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(sample_rate, 16000)(audio)

    audio = fix_length(audio, 16000)

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
        return len(self.audio_paths)*5000 


class BachViolinRoll(Dataset):
    """ Create a Dataset for Bach Violin. Each item is a tuple of the form:
    waveform, proll waveform, sample_rate"""

    def __init__(self, path: str):
        audio_path = os.path.join(path, "audio")
        proll_path = os.path.join(path, "proll")

        # find all wav files in these paths using os
    
        self.audio_paths = []
        # for filename in glob.glob(os.path.join(path, '*.wav')):

        for filename in glob.glob(os.path.join(audio_path, '*.wav')):
            self.audio_paths.append(filename)

        self.proll_paths = []
        for filename in glob.glob(os.path.join(proll_path, '*.wav')):
            self.proll_paths.append(filename)

        #print(len(self.audio_paths))
        #assert False
        assert len(self.audio_paths) > 0

        #print(len(self.audio_paths))

        assert len(self.audio_paths) == len(self.proll_paths)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        n = n % len(self.audio_paths)
        audiofile = self.audio_paths[n]
        prollfile = self.proll_paths[n]

        # load audio wave
        audio, sample_rate = torchaudio.load(audiofile)
        
        # load proll wave
        proll, sample_rate2 = torchaudio.load(prollfile)

        assert sample_rate == sample_rate2
        assert audio.shape[1] == proll.shape[1]

        # find a 1 random second segment
        audio_len = audio.shape[1]
        start = np.random.randint(0, audio_len - SAMPLE_RATE)
        audio = audio[:,start:start+SAMPLE_RATE]
        proll = proll[:,start:start+SAMPLE_RATE]

        return audio, proll, sample_rate

    def __len__(self) -> int:
        # randomly sample parts of the audio file
        # to cover all parts of the audio file
        return len(self.audio_paths)*5000