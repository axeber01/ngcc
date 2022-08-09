from torchaudio.datasets import LIBRISPEECH
import pyroomacoustics as pra
import numpy as np
from typing import Tuple
from torch import Tensor
import torch
import random
from librosa.effects import split


def remove_silence(signal, top_db=20, frame_length=2048, hop_length=512):
    '''
    Remove silence from speech signal
    '''
    signal = signal.squeeze()
    clips = split(signal, top_db=top_db,
                  frame_length=frame_length, hop_length=hop_length)
    output = []
    for ii in clips:
        start, end = ii
        output.append(signal[start:end])

    return torch.cat(output)


class LibriSpeechLocations(LIBRISPEECH):
    '''
    Class of LibriSpeech recordings. Each recording is annotated with a speaker location.
    '''

    def __init__(self, source_locs, split):
        super().__init__("./", url=split, download=True)

        self.source_locs = source_locs

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int, int, float, int]:

        source_loc = self.source_locs[n]
        seed = n
        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_number = super().__getitem__(n)
        return (waveform, sample_rate, transcript, speaker_id, utterance_number), source_loc, seed


def one_random_delay(room_dim, fs, t60, mic_locs, signal, xyz_min, xyz_max, snr, anechoic=False):
    '''
    Simulate signal propagation using pyroomacoustics using random source location.
    '''

    if anechoic:
        e_absorption = 1.0
        max_order = 0
    else:
        e_absorption, max_order = pra.inverse_sabine(t60, room_dim)

    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(
        e_absorption), max_order=max_order)

    source_loc = np.random.uniform(low=xyz_min, high=xyz_max, size=(3))
    room.add_source(source_loc, signal=signal.squeeze())
    room.add_microphone(mic_locs)
    c = room.c
    d = np.sqrt(np.sum((mic_locs[:, 0] - source_loc)**2)) - \
        np.sqrt(np.sum((mic_locs[:, 1] - source_loc)**2))
    delay = d * fs / c
    room.simulate(reference_mic=0, snr=snr)
    x1 = room.mic_array.signals[0, :]
    x2 = room.mic_array.signals[1, :]

    return x1, x2, delay, room


def one_delay(room_dim, fs, t60, mic_locs, signal, source_loc, snr=1000, anechoic=False):
    '''
    Simulate signal propagation using pyroomacoustics for a given source location.
    '''

    if anechoic:
        e_absorption = 1.0
        max_order = 0
    else:
        e_absorption, max_order = pra.inverse_sabine(t60, room_dim)

    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(
        e_absorption), max_order=max_order)

    room.add_source(source_loc, signal=signal.squeeze())
    room.add_microphone(mic_locs)
    c = room.c
    d = np.sqrt(np.sum((mic_locs[:, 0] - source_loc)**2)) - \
        np.sqrt(np.sum((mic_locs[:, 1] - source_loc)**2))
    delay = d * fs / c
    room.simulate(reference_mic=0, snr=snr)
    x1 = room.mic_array.signals[0, :]
    x2 = room.mic_array.signals[1, :]

    return x1, x2, delay, room


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0.)
    return batch


class DelaySimulator(object):
    '''
    Given a batch of LibrispeechLocation samples, simulate signal
    propagation from source to the microphone locations.
    '''

    def __init__(self, room_dim, fs, N, t60, mic_locs, max_tau, anechoic, train=True, snr=1000, lower_bound=16000, upper_bound=48000):

        self.room_dim = room_dim
        self.fs = fs
        self.N = N
        self.mic_locs = mic_locs
        self.max_tau = max_tau
        self.snr = snr
        self.t60 = t60
        self.anechoic = anechoic
        self.train = train

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, batch):
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        tensors1, tensors2, targets = [], [], []

        # Gather in lists, and encode labels as indices
        with torch.no_grad():
            for (waveform, sample_rate,  _, _, _), source_loc, seed in batch:

                waveform = waveform.squeeze()
                signal = remove_silence(waveform, frame_length=self.N)

                # use random seed for training, fixed for val/test
                # this controls the randomness in sound propagation when simulating the room
                if not self.train:
                    torch.manual_seed(seed)
                    random.seed(seed)
                    np.random.seed(seed)

                # sample random reverberation time and SNR
                this_t60 = np.random.uniform(low=self.t60[0], high=self.t60[1])
                this_snr = np.random.uniform(low=self.snr[0], high=self.snr[1])

                x1, x2, delay, _ = one_delay(room_dim=self.room_dim, fs=self.fs, t60=this_t60,
                                             mic_locs=self.mic_locs, signal=signal,
                                             source_loc=source_loc, snr=this_snr,
                                             anechoic=self.anechoic)

                if self.train:
                    start_idx = torch.randint(
                        self.lower_bound, self.upper_bound - self.N - 1, (1,))
                else:
                    start_idx = self.lower_bound

                end_idx = start_idx + self.N
                x1 = x1[start_idx:end_idx]
                x2 = x2[start_idx:end_idx]

                tensors1 += [torch.as_tensor(x1, dtype=torch.float)]
                tensors2 += [torch.as_tensor(x2, dtype=torch.float)]
                targets += [delay+self.max_tau]

        # Group the list of tensors into a batched tensor
        tensors1 = pad_sequence(tensors1).unsqueeze(1)
        tensors2 = pad_sequence(tensors2).unsqueeze(1)
        targets = torch.Tensor(targets)

        return tensors1, tensors2, targets
