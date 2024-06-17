import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch as th
from scipy.signal import find_peaks
import mir_eval
import torchaudio

SAMPLING_RATE = 44100
HOP_LENGTH = 512
ONSETS_ABS_ERROR_RATE_IN_SECONDS = 0.050
WIN_LENGTHS = [1024, 2048, 4096]
WIN_SIZES = [0.023, 0.046, 0.093]
# WIN_LENGTHS = [int(SAMPLING_RATE * 0.023), int(SAMPLING_RATE * 0.046), int(SAMPLING_RATE * 0.093)]
N_MELS = 80
F_MIN = 27.5
F_MAX = 16000
FRAME_LENGTH = 15


def load_dataset_paths(data_folder, is_train_dataset=True, count=0):
    """
    Searches for .wav files and associated annotation files in the given directory,
    compiling lists of file paths based on whether the dataset is for training.

    Parameters:
        data_folder (str): Path to the dataset directory.
        is_train_dataset (bool): Flag to include annotation files; defaults to True.

    Returns:
        tuple: Lists of paths for .wav files and, if applicable, their annotations (.beats.gt, .onsets.gt, .tempo.gt).
    """
    wav_files_paths = []
    beat_files_paths = []
    onset_files_paths = []
    tempo_files_paths = []
    i = 0
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.wav'):
                wav_file_path = os.path.join(root, file)
                wav_files_paths.append(wav_file_path)
                beat_file_path = wav_file_path.replace('.wav', '.beats.gt')

                if is_train_dataset:
                    onset_file_path = wav_file_path.replace('.wav', '.onsets.gt')
                    onset_files_paths.append(onset_file_path)

                tempo_file_path = wav_file_path.replace('.wav', '.tempo.gt')
                beat_files_paths.append(beat_file_path)
                tempo_files_paths.append(tempo_file_path)
                i += 1
                if 0 <= count == i:
                    return wav_files_paths, beat_files_paths, onset_files_paths, tempo_files_paths

    return wav_files_paths, beat_files_paths, onset_files_paths, tempo_files_paths


def make_frames(X, y, onsets, sample_rate, frame_length=FRAME_LENGTH):
    X_frames, y_frames = [], []
    half_frame_length = frame_length // 2

    for onset_time in onsets:
        onset_idx = int(convert_times_to_frames(onset_time, sample_rate))

        start = max(0, onset_idx - half_frame_length)
        end = min(onset_idx + half_frame_length + 1, X.shape[2] - frame_length)

        idx = start
        while idx < end:
            X_frame = X[:, :, idx:idx+frame_length]
            center_idx = idx + half_frame_length

            if y[center_idx] == 1:
                y_label = 1
            else:
                y_label = 0

            X_frames.append(X_frame)
            y_frames.append(y_label)
            idx += 1

    return X_frames, y_frames


def preprocess_audio(files):
    spectrograms = []
    sample_rates = []

    for file_path in tqdm(files):
        waveform, sample_rate = torchaudio.load(file_path)
        mel_specgram = calculate_melbands(waveform[0], sample_rate)
        spectrograms.append(mel_specgram)
        sample_rates.append(sample_rate)

    return spectrograms, sample_rates



def calculate_melbands(waveform, sample_rate):
    mel_specs = []
    for wl in WIN_LENGTHS:
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=wl,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX
        )(waveform)
        mel_specs.append(mel_spectrogram)
    return th.log10(th.stack(mel_specs) + 1e-08)


def load_onsets(onset_paths: list):
    onsets = []
    for onset_path in onset_paths:
        with open(onset_path, 'r') as f:
            o = list(map(float, f.read().split()))
        onsets.append(np.array(o))
    return onsets


def load_tempo_annotations(file_path):
    annotations = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) == 1:
                # Case with one tempo
                tempo = float(parts[0])
                annotations.extend(tempo)
                annotations.extend(1.0)
            elif len(parts) == 3:
                annotations.extend(parts)
            else:
                raise ValueError(f"Unexpected format in line: {line}")

    return {"tempo": annotations}


def load_tempo_annotations_from_files(file_list: list):
    annotations = {}

    for file_path in file_list:
        filename = os.path.basename(file_path).replace('.tempo.gt', '')
        annotations[filename] = load_tempo_annotations(file_path)

    return annotations


def make_target(onsets, no_of_frames, sample_rate):
    y = np.zeros(no_of_frames)
    for onset in onsets:
        onset_frame = int(convert_times_to_frames(onset, sample_rate))
        y[onset_frame] = 1
    return y


def convert_times_to_frames(onset_time, sample_rate):
    return onset_time * sample_rate / HOP_LENGTH


def convert_frames_to_times(onset_frame, sample_rate):
    return onset_frame * HOP_LENGTH / sample_rate


def convert_onset_frames_to_times(onset_signal, sample_rate):
    res = []
    for idx, value in enumerate(onset_signal):
        if (value > 0):
            res.append(idx * HOP_LENGTH / sample_rate)
    return np.array(res)


def make_frames(spectrogram, onsets, sample_rate):
    num_window_sizes, n_mels, num_frames = spectrogram.shape
    half_context = FRAME_LENGTH // 2
    X_frames = []
    y_frames = []

    onset_frames = (onsets * sample_rate / HOP_LENGTH).astype(int)

    for onset_frame in onset_frames:
        for i in [-2, -1, 0, 1, 2]:
            frame_idx = onset_frame + i
            if frame_idx - half_context < 0 or frame_idx + half_context >= num_frames:
                continue

            frame = spectrogram[:, :, frame_idx - half_context:frame_idx + half_context + 1]
            X_frames.append(th.tensor(frame, dtype=th.float32))

            # Label is 1 if the onset is in the middle, otherwise 0
            label = 1 if i == 0 else 0
            y_frames.append(th.tensor([label], dtype=th.float32))

    return X_frames, y_frames


def binary_predictions_to_times(binary_preds, sample_rate):
    onset_frames = np.where(binary_preds == 1)[0]
    onset_times = onset_frames * HOP_LENGTH / sample_rate
    return onset_times


def moving_average_smoothing(predictions, window_size):
    cumsum_vec = np.cumsum(np.insert(predictions, 0, 0))
    smoothed = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return smoothed


def evaluate_model(model, features, sample_rates, data_mean, data_std, threshold=0.5):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    model.eval()
    all_preds = []

    for idx, x_test in enumerate(features):
        x_test = x_test.to(device)
        x_test = (x_test - data_mean) / data_std
        num_frames = x_test.shape[-1]
        frames = []

        with th.no_grad():
            for i in range(0, num_frames - FRAME_LENGTH + 1):
                frame = x_test[:, :, i:i + FRAME_LENGTH].unsqueeze(0)
                frames.append(frame)

            frames = th.cat(frames, dim=0)
            outputs = model(frames).squeeze().cpu().numpy()
            binary_preds = (outputs >= threshold).astype(int)
            pred_times = binary_predictions_to_times(binary_preds, sample_rates[idx])
            all_preds.append(pred_times)

    return all_preds


#################### Utils for Tempo ########################

# This is the almost the same as the prediction function in the onset detection but here we just use the onset signal
# and no prediction
def raw_onset_signal(model, x, mean=mean, std=std, frame_size=15):
    model = model.to(device)
    model.eval()
    x = x.to(device)
    mean = mean.to(device)
    std = std.to(device)
    x = (x - mean) / std

    half_frame_size = frame_size // 2
    num_frames = x.shape[2]
    onset_predictions = []

    with th.no_grad():
        for j in range(half_frame_size, num_frames - half_frame_size):
            start = j - half_frame_size
            end = j + half_frame_size + 1
            input_frame = x[:, :, start:end].unsqueeze(0).float()
            output = model(input_frame).squeeze().cpu().item()
            onset_predictions.append(output)
    onset_predictions = np.array(onset_predictions)
    onset_signal = np.convolve(onset_predictions, np.hamming(10), mode='same')
    return onset_signal

def autocorrelate(signal, lag):
    r = np.zeros(len(signal) - lag)
    for t in range(len(signal) - lag):
        r[t] = signal[t + lag] * signal[t]
    return np.sum(r)

def to_bpm(max_r):
    return 60 * SAMPLING_RATE / HOP_LENGTH / (max_r + 25)

def autocorrelate_tao(signal, min_tao=25, max_tao=87):
    return np.array([autocorrelate(signal, tao) for tao in range(min_tao, max_tao)])

def get_tempo(model, x, top_n=2):
    onset_signal_res = raw_onset_signal(model, x)
    taos = autocorrelate_tao(onset_signal_res)
    peaks = find_peaks(taos)[0]
    highest_peaks = np.argsort(-taos[peaks])[:top_n]

    return list(reversed([to_bpm(r) for r in peaks[highest_peaks]]))
