import os
import librosa
import numpy as np
import pandas as pd

SAMPLING_RATE = 44100
HOP_LENGTH = 512
ONSETS_ABS_ERROR_RATE_IN_SECONDS = 0.050


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
    wav_files_paths, beat_files_paths, onset_files_paths, tempo_files_paths = load_dataset_paths(data_folder, True)
    df = pd.DataFrame({'File Path': wav_files_paths})
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
