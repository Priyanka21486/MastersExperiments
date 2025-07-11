# -*- coding: utf-8 -*-
"""chatgpt.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CwPvknzAf6-Czhp82ORb_dJgg0uISjOh
"""

import scipy.io as sio

def inspect_mat_file(file_path):
    data = sio.loadmat(file_path)
    print("Keys in the .mat file:", data.keys())
    for key in data:
        print(f"Key: {key}, Type: {type(data[key])}, Shape: {data[key].shape if hasattr(data[key], 'shape') else 'N/A'}")

# Example usage
syllable_file = '/content/Syllable.mat'
inspect_mat_file(syllable_file)

import numpy as np
import scipy.io as sio
import scipy.signal as signal
import librosa

def vocoder_func(wavPath):
    # Load the audio file
    y, Fs = librosa.load(wavPath, sr=None)

    # Parameters for the vocoder
    fltFc = np.array([240, 360, 480, 600, 720, 840, 1000, 1150, 1300, 1450, 1600, 1800, 2000, 2200, 2400, 2700, 3000, 3300, 3750])
    fltBW = np.array([120, 120, 120, 120, 120, 120, 150, 150, 150, 150, 150, 200, 200, 200, 200, 300, 300, 300, 500])
    fltLpFc = 3000
    numBands = len(fltFc)

    # Bandpass filter function
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.lfilter(b, a, data)

    # Lowpass filter function
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low')
        return signal.lfilter(b, a, data)

    # Frame the signal
    frameLength = 2048
    overlap = 1024
    frames = np.arange(0, len(y) - frameLength, overlap)
    sigFrames = [y[int(start):int(start + frameLength)] for start in frames]

    sub_band_energies = []

    for j in range(len(sigFrames)):
        currFrame = sigFrames[j]
        frame_energies = []
        for i in range(numBands):
            center_freq = fltFc[i]
            bandwidth = fltBW[i]
            lowcut = center_freq - bandwidth / 2
            highcut = center_freq + bandwidth / 2
            fltFrame = butter_bandpass_filter(currFrame, lowcut, highcut, Fs, 2)
            rectFrame = np.abs(fltFrame)
            lpFltFrame = butter_lowpass_filter(rectFrame, fltLpFc, Fs, 2)
            frame_energies.append(np.mean(lpFltFrame))
        sub_band_energies.append(frame_energies)

    return np.array(sub_band_energies)

def compute_stress_features(wav_file, syllable_file):
    # Load syllable information
    syllable_data = sio.loadmat(syllable_file)
    spurt_syl = syllable_data['spurtSyl'].flatten()  # Syllable labels
    spurt_syl_times = syllable_data['spurtSylTimes']  # Times for syllables

    # Process the wav file with vocoder
    sub_band_energies = vocoder_func(wav_file)

    # Extract features
    features = {}
    for i, syl in enumerate(spurt_syl):
        syl_start = int(spurt_syl_times[i, 0])
        syl_end = int(spurt_syl_times[i, 1])

        # Ensure that the syllable indices are within bounds
        if syl_start >= len(sub_band_energies) or syl_end >= len(sub_band_energies) or syl_start >= syl_end:
            features[syl] = np.full((20,), np.nan)  # or some other way to indicate error
            continue

        syl_energies = sub_band_energies[syl_start:syl_end]

        # Ensure that syllable energies are not empty
        if len(syl_energies) == 0:
            features[syl] = np.full((20,), np.nan)  # or some other way to indicate error
        else:
            features[syl] = np.mean(syl_energies, axis=0)  # Example: mean energy for the syllable

    return features

# Example usage
wav_file = '/content/ISLE_SESS0006_BLOCKD01_06_sprt1.wav'
syllable_file = '/content/Syllable.mat'
stress_features = compute_stress_features(wav_file, syllable_file)
print(stress_features)

import numpy as np
import scipy.io as sio
import scipy.signal as signal
import librosa

def vocoder_func(wavPath):
    # Load the audio file
    y, Fs = librosa.load(wavPath, sr=None)

    # Parameters for the vocoder
    fltFc = np.array([240, 360, 480, 600, 720, 840, 1000, 1150, 1300, 1450, 1600, 1800, 2000, 2200, 2400, 2700, 3000, 3300, 3750])
    fltBW = np.array([120, 120, 120, 120, 120, 120, 150, 150, 150, 150, 150, 200, 200, 200, 200, 300, 300, 300, 500])
    fltLpFc = 3000
    numBands = len(fltFc)

    # Bandpass filter function
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.lfilter(b, a, data)

    # Lowpass filter function
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low')
        return signal.lfilter(b, a, data)

    # Frame the signal
    frameLength = 2048
    overlap = 1024
    frames = np.arange(0, len(y) - frameLength, overlap)
    sigFrames = [y[int(start):int(start + frameLength)] for start in frames]

    sub_band_energies = []

    for j in range(len(sigFrames)):
        currFrame = sigFrames[j]
        frame_energies = []
        for i in range(numBands):
            center_freq = fltFc[i]
            bandwidth = fltBW[i]
            lowcut = center_freq - bandwidth / 2
            highcut = center_freq + bandwidth / 2
            fltFrame = butter_bandpass_filter(currFrame, lowcut, highcut, Fs, 2)
            rectFrame = np.abs(fltFrame)
            lpFltFrame = butter_lowpass_filter(rectFrame, fltLpFc, Fs, 2)
            frame_energies.append(np.mean(lpFltFrame))
        sub_band_energies.append(frame_energies)

    return np.array(sub_band_energies)

def compute_stress_features(wav_file, syllable_file):
    # Load syllable information
    syllable_data = sio.loadmat(syllable_file)
    spurt_syl = syllable_data['spurtSyl'].flatten()  # Syllable labels
    spurt_syl_times = syllable_data['spurtSylTimes']  # Times for syllables

    # Process the wav file with vocoder
    sub_band_energies = vocoder_func(wav_file)

    # Extract features
    features = {}
    for i, syl in enumerate(spurt_syl):
        syl_start_1 =(spurt_syl_times[i, 0])
        syl_end_1 = (spurt_syl_times[i, 1])
        syl_start = int(np.round(syl_start_1 * 16000))
        syl_end = int(np.round(syl_end_1 * 16000))

        # Debugging information
        print(f"Syllable: {syl}")
        print(f"Start frame: {syl_start_1}, End frame: {syl_end_1}")

        # Ensure that the syllable indices are within bounds
        if syl_start_1 >= len(sub_band_energies) or syl_end_1 >= len(sub_band_energies) or syl_start >= syl_end:
            features[syl] = np.full((20,), np.nan)  # or some other way to indicate error
            print(f"Out of bounds: {syl}")
            continue

        syl_energies = sub_band_energies[syl_start:syl_end]

        # Ensure that syllable energies are not empty
        if len(syl_energies) == 0:
            features[syl] = np.full((20,), np.nan)  # or some other way to indicate error
            print(f"Empty segment: {syl}")
        else:
            features[syl] = np.mean(syl_energies, axis=0)  # Example: mean energy for the syllable

    return features

# Example usage
wav_file = '/content/ISLE_SESS0006_BLOCKD01_06_sprt1.wav'
syllable_file = '/content/Syllable.mat'
stress_features = compute_stress_features(wav_file, syllable_file)
print(stress_features)

import scipy.io as sio

def print_syllable_mat_contents(file_path):
    # Load the .mat file
    data = sio.loadmat(file_path)

    # Print the keys of the .mat file
    print("Keys in the .mat file:", data.keys())

    # Print the content of each key
    for key in data:
        if key.startswith('__'):
            continue  # Skip metadata keys
        print(f"Key: {key}, Type: {type(data[key])}, Shape: {data[key].shape if hasattr(data[key], 'shape') else 'N/A'}")
        print(data[key])

# Example usage
syllable_file = '/content/Syllable.mat'
print_syllable_mat_contents(syllable_file)

