import os
import librosa
import numpy as np

SAMPLING_RATE = 44100


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
