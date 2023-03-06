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


MAX_WAV_VALUE = 32768.0

def files_to_list(data_path):
    """
    Load all .wav files in data_path
    """
    # Go through all subdirectories and find the files
    ret = []
    files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.mp3'):
                files.append(file)
                ret.append(os.path.join(root, file))
    return ret, files

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
    Create a Dataset for Speech Commands. Each item is a tuple of the form:
    waveform, sample_rate, label
    """

    def __init__(self, path: str):
        self.audio_paths, self.audio_files = files_to_list(path) 

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        n = n % len(self.audio_paths)
        filename = self.audio_paths[n]
        return load_wav_to_torch(filename)

    def __len__(self) -> int:
        # randomly sample parts of the audio file
        # to cover all parts of the audio file
        return len(self.audio_paths)*5000 

class BachViolinWithPianoRoll(Dataset):
    """
    Create a Dataset for Speech Commands. Each item is a tuple of the form:
    waveform, sample_rate, label
    """

    def __init__(self, path: str):
        
        

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        n = n % len(self.audio_paths)
        filename = self.audio_paths[n]
        audio, sample_rate, label = load_wav_to_torch(filename)
        piano_roll = self.get_piano_roll(audio)
        return audio, sample_rate, label, piano_roll

    def __len__(self) -> int:
        # randomly sample parts of the audio file
        # to cover all parts of the audio file
        return len(self.audio_paths)*5000 
    
    def load_notes(self, filename):
        """
        Load csv files corresponding to every audio file
        The files are named the same as the audio file except with .csv extension
        It is in the notes folder instead of the audio folder
        The format of the csv file is:
        onset,offset,pitch,velocity
        """
        
        # find all the csv files in the notes folder in the order of the audio files
        # the audio files are in self.audio_files = files_to_list(path)

        # walk through the notes folder
        # and find the csv files in the order of the audio files
        # and load them into a list
        # the list is the same length as self.audio_files
        # and each element is a numpy array of shape (n, 4)
        # where n is the number of notes in the csv file
        # and the 4 columns are onset, offset, pitch, velocity
        
        ret = []
        for root, dirs, files in os.walk(filename):
            for file in files:
                if file.endswith('.csv'):
                    ret.append(os.path.join(root, file))

        # arrange the csv files in the same order as the audio files
        ret = sorted(ret, key=lambda x: self.audio_files.index(x[:-4]+'.mp3'))

        # load the csv files into a list of numpy arrays
        ret = [np.loadtxt(x, delimiter=',') for x in ret]

        return ret
    
    def load_alignment(self, filename):
        """
        Load csv files corresponding to every audio file
        The files are named the same as the audio file except with .csv extension
        It is in the alignment folder instead of the audio folder
        The format of the csv file is:
        start,end
        """
        # Same as load_notes but for the alignment files

        ret = []
        for root, dirs, files in os.walk(filename):
            for file in files:
                if file.endswith('.csv'):
                    ret.append(os.path.join(root, file))

        # arrange the csv files in the same order as the audio files
        ret = sorted(ret, key=lambda x: self.audio_files.index(x[:-4]+'.mp3'))

        # load the csv files into a list of numpy arrays
        ret = [np.loadtxt(x, delimiter=',') for x in ret]

        return ret
    
    def get_piano_roll(self, i):


        # alignment is a list of numpy arrays of shape (n, 2)
        # where n is the number of notes in the csv file
        # and the 2 columns are start, end time of the notes in the audio file

        # create a piano roll for each audio file, 
        # the piano roll is going to be the same size as the audio file
        
        """
        Sample code to create a piano roll segment

        global_end = round(
        max(float(alignment[i]["end"]) for i in csv_row_indices)
            / args.resolution
        )
        global_start = round(
            float(alignment[csv_row_indices[0]]["start"]) / args.resolution
        )
        pianoroll = np.zeros((global_end - global_start, 128), bool)
        for csv_row_idx in csv_row_indices:
            csv_row = alignment[csv_row_idx]
            start_ = (
                round(float(csv_row["start"]) / args.resolution)
                - global_start
            )
            end_ = (
                round(float(csv_row["end"]) / args.resolution)
                - global_start
            )
            pitch = int(notes[csv_row_idx]["pitch"])
            pianoroll[start_:end_, pitch] = 1
        start_idx = rounded_start - global_start
        end_idx = rounded_end - global_start
                
        """
        # create corresponding piano roll for each audio file

        notes = self.load_notes(None)[i]
        alignment = self.load_alignment(None)[i]

        # create a piano roll for each audio file,
        # the piano roll is going to be the same size as the audio file

        for i in 





        





        

    
    def load_alignment(self, filename):
        return np.load(filename)

    def get_piano_roll(self, audio):