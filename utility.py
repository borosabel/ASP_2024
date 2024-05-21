import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch as th

SAMPLING_RATE = 44100
HOP_LENGTH = 512
ONSETS_ABS_ERROR_RATE_IN_SECONDS = 0.050
WIN_LENGTHS = [1024, 2048, 4096]
# WIN_LENGTHS = [int(SAMPLING_RATE * 0.023), int(SAMPLING_RATE * 0.046), int(SAMPLING_RATE * 0.093)]
N_MELS = 80
F_MIN = 27.5
F_MAX = 16000
FRAME_LENGTH = 15


def load_dataset_paths(data_folder, is_train_dataset=True):
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

    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.wav'):
                wav_file_path = os.path.join(root, file)
                wav_files_paths.append(wav_file_path)

                if is_train_dataset:
                    beat_file_path = wav_file_path.replace('.wav', '.beats.gt')
                    onset_file_path = wav_file_path.replace('.wav', '.onsets.gt')
                    tempo_file_path = wav_file_path.replace('.wav', '.tempo.gt')
                    beat_files_paths.append(beat_file_path)
                    onset_files_paths.append(onset_file_path)
                    tempo_files_paths.append(tempo_file_path)

    return wav_files_paths, beat_files_paths, onset_files_paths, tempo_files_paths

def preprocess_audio(files):
    spectrograms = []
    sample_rates = []

    for file_path in tqdm(files):
        waveform, sample_rate = librosa.load(file_path, sr=None)
        waveform = librosa.util.normalize(waveform)
        mel_specgram = calculate_melbands(waveform, sample_rate)
        spectrograms.append(mel_specgram)
        sample_rates.append(sample_rate)

    return spectrograms, sample_rates


def calculate_melbands(waveform, sample_rate):
    mel_specs = []
    for wl in WIN_LENGTHS:
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=wl,
            hop_length=HOP_LENGTH,
            n_mels=80,
            fmin=27.5,
            fmax=16000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_specs.append(mel_spec_db)
    return np.stack(mel_specs)


def load_onsets(onset_paths: list):
    onsets = []
    for onset_path in onset_paths:
        with open(onset_path, 'r') as f:
            o = list(map(float, f.read().split()))
        onsets.append(np.array(o))
    return onsets


def make_target(onsets, no_of_frames, sample_rate):
    y = np.zeros(no_of_frames)
    for onset in onsets:
        x_i = int(convert_onset_times_to_frames(onset, sample_rate))
        y[x_i] = 1
    return y


def convert_onset_times_to_frames(onset_time, sample_rate):
    return onset_time * sample_rate / HOP_LENGTH


def convert_onset_frames_to_times(onset_signal, sample_rate):
    res = []
    for idx, value in enumerate(onset_signal):
        if(value > 0):
            res.append(idx * HOP_LENGTH / sample_rate)
    return np.array(res)

def make_frames(spectrogram, labels, onset_times, sample_rate, frame_length=15):
    X_frames = []
    y_frames = []

    # Generate positive frames (frames with onset)
    for onset_time in onset_times:
        onset_idx = int(convert_onset_times_to_frames(onset_time, sample_rate))

        start = max(0, onset_idx - frame_length // 2)
        end = min(onset_idx + frame_length // 2 + 1, spectrogram.shape[2])

        X_frame = spectrogram[:, :, start:end]
        if X_frame.shape[2] < frame_length:
            pad_width = frame_length - X_frame.shape[2]
            X_frame = np.pad(X_frame, ((0, 0), (0, 0), (0, pad_width)), mode='constant')

        X_frames.append(X_frame)
        y_frames.append(1 if labels[start:end].sum() > 0 else 0)

    # Generate negative frames (frames without onset)
    num_positive_samples = len(y_frames)
    all_possible_indices = np.arange(spectrogram.shape[2] - frame_length + 1)
    non_onset_indices = [idx for idx in all_possible_indices if np.sum(labels[idx:idx+frame_length]) == 0]

    np.random.shuffle(non_onset_indices)
    negative_samples = non_onset_indices[:num_positive_samples]

    for idx in negative_samples:
        start = idx
        end = start + frame_length
        X_frame = spectrogram[:, :, start:end]

        if X_frame.shape[2] < frame_length:
            pad_width = frame_length - X_frame.shape[2]
            X_frame = np.pad(X_frame, ((0, 0), (0, 0), (0, pad_width)), mode='constant')

        X_frames.append(X_frame)
        y_frames.append(0)

    # Shuffle the frames and labels together
    combined = list(zip(X_frames, y_frames))
    np.random.shuffle(combined)
    X_frames[:], y_frames[:] = zip(*combined)

    return X_frames, y_frames

# def make_frames(spectrogram, labels, onset_times, sample_rate):
#     FRAME_LENGTH = 15
#     X_frames = []
#     y_frames = []
#
#     for onset_time in onset_times:
#         onset_idx = int(convert_onset_times_to_frames(onset_time, sample_rate))
#
#         start = max(0, onset_idx - FRAME_LENGTH // 2)
#         end = min(onset_idx + FRAME_LENGTH//2 + 1, spectrogram.shape[2] - FRAME_LENGTH)
#
#         X_frames.append(spectrogram[:, :, start:end])
#         y_frames.append(1 if labels[start:end].sum() > 0 else 0)
#
#     return X_frames, y_frames


def make_test_frames(x_test, frame_length=15):
    X_frames = []
    num_frames = x_test.shape[2]

    for i in range(0, num_frames):
        if i + frame_length <= num_frames:
            X_frame = x_test[:, :, i:i+frame_length]
        else:
            # Pad the last frame with zeros
            pad_amount = i + frame_length - num_frames
            X_frame = th.nn.functional.pad(x_test[:, :, i:], (0, pad_amount))

        X_frames.append(X_frame)

    return X_frames

def create_audio_onset_dataset(spectograms, sample_rates, targets, sample_onsets):
    X_frames_list = []
    y_frames_list = []

    for X, sample_rate, y, onsets in zip(spectograms, sample_rates, targets, sample_onsets):
        X_frames, y_frames = make_frames(X, y, onsets, sample_rate)
        X_frames_list.append(X_frames)
        y_frames_list.append(y_frames)

    X_frames = np.concatenate(X_frames_list, axis=0)
    y_frames = np.concatenate(y_frames_list, axis=0)

    mean = np.mean(X_frames, axis=(0, 2), keepdims=True)
    std = np.std(X_frames, axis=(0, 2), keepdims=True)

    X_frames = (X_frames - mean) / std

    return X_frames, y_frames




















def process_audio(row):
    audio_path = row['File Path']
    onsets = row['Onsets']
    y, sr = librosa.load(audio_path, sr=SAMPLING_RATE)

    # Calculate frame length for 100ms frames
    frame_length = int(0.1 * SAMPLING_RATE)
    hop_length = frame_length  # No overlap

    # Split audio into frames
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

    # Time for each frame
    frame_times = librosa.frames_to_time(range(frames.shape[1]), sr=SAMPLING_RATE, hop_length=hop_length)

    # Initialize onset frame labels
    onset_labels = np.zeros(frames.shape[1], dtype=int)

    # Encode onsets in frame format
    for onset in onsets:
        # Find the closest frame index to each onset
        frame_idx = np.argmin(np.abs(frame_times - onset))
        onset_labels[frame_idx] = 1

    return pd.Series({
        'frame_times': frame_times,
        'onset_labels': onset_labels
    })


def get_audio_and_onsets_in_dataframe(data_folder, is_train_dataset=True):
    wav_files_paths, beat_files_paths, onset_files_paths, tempo_files_paths = load_dataset_paths(data_folder, is_train_dataset)
    df = pd.DataFrame({'File Path': wav_files_paths})
    if is_train_dataset:
        onsets_seconds = []
        for item in onset_files_paths:

            onset_seconds = []
            with open(item, 'r') as file:
                for line in file:
                    onset_time = float(line.strip())
                    onset_seconds.append(onset_time)

            onsets_seconds.append(onset_seconds)
        df['Onsets'] = onsets_seconds


    return df


def high_frequency_content(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLING_RATE)

    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    energy = np.abs(stft) ** 2
    num_bins = energy.shape[0]
    weights = np.linspace(1, 2, num_bins)
    weighted_energy = energy * weights[:, np.newaxis]

    # High frequency content
    d_HFC = np.sum(weighted_energy, axis=0) / num_bins

    # Differences between the neighboring frames
    diff_d_HFC = np.diff(d_HFC)
    threshold = np.mean(diff_d_HFC) + 1.5 * np.std(diff_d_HFC)
    onset_times = np.where(diff_d_HFC > threshold)[0]
    return librosa.frames_to_time(onset_times, sr=sr, hop_length=hop_length)

