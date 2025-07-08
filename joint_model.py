import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
from torch import nn
from torch.optim import Adam
import numpy as np
import librosa
import os
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array
import numpy as np
import librosa
import gammatone.filters
from scipy.signal import hilbert
import matplotlib.pylab as pl
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Define MFCC extraction
sample_rate = 16000
n_mfcc = 13
mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)

# Placeholder for syllable boundary detection function
def get_syllable_boundaries(utterance_path):
    def peakdet(v, delta, x=None):
        maxtab = []
        mintab = []

        if x is None:
            x = np.arange(len(v))

        v = np.asarray(v)
        if len(v) != len(x):
            raise ValueError("Input vectors v and x must have the same length")

        if not np.isscalar(delta):
            raise ValueError("Input argument delta must be a scalar")

        if delta <= 0:
            raise ValueError("Input argument delta must be positive")

        mn, mx = np.Inf, -np.Inf
        mnpos, mxpos = np.NaN, np.NaN

        lookformax = True
        for i in np.arange(len(v)):
            this = v[i]
            if this > mx or np.isnan(mx):
                mx = this
                mxpos = x[i]
            if this < mn or np.isnan(mn):
                mn = this
                mnpos = x[i]

            if lookformax:
                if this < mx - delta:
                    # Convert to scalar before appending
                    maxtab.append((mxpos, mx.item()))
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn + delta:
                    # Convert to scalar before appending
                    mintab.append((mnpos, mn.item()))
                    mx = this
                    mxpos = x[i]
                    lookformax = True

        # Convert lists to numpy arrays
        maxtab = np.array(maxtab)
        mintab = np.array(mintab)

        return maxtab, mintab

# Define the theta oscillator function
    def thetaOscillator(ENVELOPE, f=5, Q=0.5, thr=0.025, verbose=1):
        N = 10  # How many most energetic bands to use (default = 8)

        if N > ENVELOPE.size:
            print('WARNING: Input dimensionality smaller than the N parameter. Using all frequency bands.')

        a = np.array([
            [72, 34, 22, 16, 12, 9, 8, 6, 5, 4, 3, 3, 2, 2, 1, 0, 0, 0, 0, 0],
            [107, 52, 34, 25, 19, 16, 13, 11, 10, 9, 8, 7, 6, 5, 5, 4, 4, 4, 3, 3],
            [129, 64, 42, 31, 24, 20, 17, 14, 13, 11, 10, 9, 8, 7, 7, 6, 6, 5, 5, 4],
            [145, 72, 47, 35, 28, 23, 19, 17, 15, 13, 12, 10, 9, 9, 8, 7, 7, 6, 6, 5],
            [157, 78, 51, 38, 30, 25, 21, 18, 16, 14, 13, 12, 11, 10, 9, 8, 8, 7, 7, 6],
            [167, 83, 55, 41, 32, 27, 23, 19, 17, 15, 14, 12, 11, 10, 10, 9, 8, 8, 7, 7],
            [175, 87, 57, 43, 34, 28, 24, 21, 18, 16, 15, 13, 12, 11, 10, 9, 9, 8, 8, 7],
            [181, 90, 59, 44, 35, 29, 25, 21, 19, 17, 15, 14, 13, 12, 11, 10, 10, 9, 8, 8],
            [187, 93, 61, 46, 36, 30, 25, 22, 19, 17, 16, 14, 13, 12, 11, 10, 10, 9, 8, 8],
            [191, 95, 63, 47, 37, 31, 26, 23, 20, 18, 16, 15, 13, 12, 11, 11, 10, 9, 9, 8]
        ])

        i1 = max(0, min(10, round(Q * 10)))
        i2 = max(0, min(20, round(f)))

        delay_compensation = a[i1-1][i2-1]

        # Get oscillator mass
        T = 1./f  # Oscillator period
        k = 1     # Fix spring constant k = 1, define only mass
        b = 2*np.pi/T
        m = k/b**2  # Mass of the oscillator

        # Get oscillator damping coefficient
        c = np.sqrt(m*k)/Q

        # if verbose:
        #     print('Oscillator Q-value: %0.4f, center frequency: %0.1f Hz, bandwidth: %0.1f Hz.\n' % (Q, 1/T, 1/T/Q))

        # Do zero padding
        e = np.transpose(ENVELOPE)
        e = np.vstack((e, np.zeros((500, e.shape[1]))))
        F = e.shape[1]  # Number of frequency channels

        # Get oscillator amplitudes as a function of time
        x = np.zeros((e.shape[0], F))
        a = np.zeros((e.shape[0], F))
        v = np.zeros((e.shape[0], F))

        for t in range(1, e.shape[0]):
            for cf in range(F):
                f_up = e[t, cf]  # driving positive force
                f_down = -k * x[t-1, cf] - c * v[t-1, cf]
                f_tot = f_up + f_down  # Total force
                # Get acceleration from force
                a[t, cf] = f_tot/m

                # Get velocity from acceleration
                v[t, cf] = v[t-1, cf] + a[t, cf] * 0.001  # assumes 1000 Hz sampling
                # Get position from velocity
                x[t, cf] = x[t-1, cf] + v[t, cf] * 0.001

        # Perform group delay correction by removing samples from the
        # beginning and adding zeroes to the end
        for f in range(F):
            if delay_compensation:
                x[:, f] = np.append(x[delay_compensation:, f], np.zeros((delay_compensation, 1)))

        x = x[:-500]  # Remove zero-padding

        # Combine N most energetic bands to get sonority envelope
        tmp = x
        tmp = tmp - np.min(tmp) + 0.00001
        x = np.zeros((tmp.shape[0], 1))

        for zz in range(tmp.shape[0]):
            sort_tmp = np.sort(tmp[zz, :], axis=0)[::-1]
            x[zz] = sum((np.log10(sort_tmp[:N])))

        # Scale sonority envelope between 0 and 1
        x = x - np.min(x)
        x = x / np.max(x)
        return x

# Generate Gammatone filterbank center frequencies (log-spacing)
    minfreq = 50
    maxfreq = 7500
    bands = 20

    cfs = np.zeros((bands, 1))
    const = (maxfreq/minfreq)**(1/(bands-1))

    cfs[0] = 50
    for k in range(bands-1):
        cfs[k+1] = cfs[k] * const

    # Read the audio data
    wav_data, fs = librosa.load(utterance_path)
    wav_data = librosa.resample(y=wav_data, orig_sr=fs, target_sr=16000)
    fs = 16000
    # Compute gammatone envelopes and downsample to 1000 Hz
    coefs = gammatone.filters.make_erb_filters(fs, cfs, width=1.0)
    filtered_signal = gammatone.filters.erb_filterbank(wav_data, coefs)
    hilbert_envelope = np.abs(hilbert(filtered_signal))
    env = librosa.resample(y=hilbert_envelope, orig_sr=fs, target_sr=1000)

    # Run oscillator-based segmentation
    Q_value = 0.5  # Q-value of the oscillator, default = 0.5 = critical damping
    center_frequency = 5  # in Hz
    threshold = 0.01

    # Get the sonority function
    outh = thetaOscillator(env, center_frequency, Q_value, threshold)

    # Detect the peaks and valleys of the sonority function
    peaks, valleys = peakdet(outh, threshold)
    syllable_timestamps = []

    if len(valleys) and len(peaks):
        valley_indices = valleys[:, 0]
        peak_indices = peaks[:, 0]

        # Add signal onset if not detected by valley picking
        if valley_indices[0] > 50:
            valley_indices = np.insert(valley_indices, 0, 0)
        if valley_indices[-1] < env.shape[1] - 50:
            valley_indices = np.append(valley_indices, env.shape[1])
    else:
        valley_indices = [0, len(env)]


    #Made changes here to retrienve the timestamps
    #-------------------------------------------------------------------------------
    # Collect the start and end times of each syllable
    for i in range(len(valley_indices) - 1):
        start_time = valley_indices[i] / 1000.0
        end_time = valley_indices[i + 1] / 1000.0
        syllable_timestamps.append((start_time, end_time))
    #-------------------------------------------------------------------------------


   
    return(syllable_timestamps)









# def get_syllable_boundaries(utterance_path):
#     # Implement your logic here to return start and end indices of syllables
#     # For demonstration, returning dummy data
#     return [(0, 100), (101, 200)]  # Dummy syllable boundaries

# Step 2: Preprocess MFCCs according to syllable boundaries
def segment_mfcc_by_syllables(waveform_np,syllable_boundaries,sr):
    segments = []
        
    # return segments
    for start, end in syllable_boundaries:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        print(start_sample, end_sample)
        syllable_audio = waveform_np[start_sample:end_sample]
        print((syllable_audio))
        print(len)
        # mfcc_features_list = []
        if len(syllable_audio) >= 1024:
                mfcc = librosa.feature.mfcc(y=syllable_audio, sr=sr, n_fft=1024, n_mfcc=13)
                print(f"mfcc = {mfcc}")
                # segments.append(mfcc)
                # mfcc_features_list.append(mfcc_features)
                # print(f"Syllable {i+1} MFCC features shape: {mfcc_features.shape}")
        else:
                print(f"Syllable is too short for MFCC computation")
                # print(f"syll audio type = {type(syllable_audio)}")
        # syllable_audio = np.array(syllable_audio)

        # mfcc = librosa.feature.mfcc(y=syllable_audio, sr=sr, n_fft=1024, n_mfcc=13)
        # print(f"mfcc = {mfcc}")
        segments.append(mfcc)
    return segments
        
def average_mfcc_segments(segments):
    averaged_segments = [segment.mean(dim=0) for segment in segments]
    return torch.stack(averaged_segments)

# Step 3: Custom Dataset Class
class AudioMFCCDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.mfcc_data = []
        self.labels = []

        # Assuming each subdirectory in root_dir represents a class
        class_dirs = os.listdir(root_dir)
        for label, class_name in enumerate(class_dirs):
            class_path = os.path.join(root_dir, class_name)
            for audio_file in os.listdir(class_path):
                audio_path = os.path.join(class_path, audio_file)
                waveform, _ = torchaudio.load(audio_path)
                waveform_np = waveform.numpy()

                # mfcc = mfcc_transform(waveform)
                syllable_boundaries = get_syllable_boundaries(audio_path)
                segments = segment_mfcc_by_syllables(waveform_np, syllable_boundaries,sr=16000)
                # averaged_segments = average_mfcc_segments(segments)
                self.mfcc_data.append(segments)
                self.labels.append(label)

    def __len__(self):
        return len(self.mfcc_data)

    def __getitem__(self, idx):
        return self.mfcc_data[idx], self.labels[idx]

# Step 4: Model Architecture
class SyllableClassifier(nn.Module):
    def __init__(self):
        super(SyllableClassifier, self).__init__()
        self.bilstm = nn.LSTM(input_size=n_mfcc, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True)
        self.tdnn = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, x.size(2), n_mfcc)
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        x = x.view(-1, x.size(1), 128)
        x = self.tdnn(x)
        return x.squeeze(1)

# Step 5: Training Loop Setup
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Loss: {epoch_loss:.4f}')

# Main execution
root_dir = '/home/spl_cair/Desktop/priyanka/icassp_exp/jointdirectory'  # Change this path to your directory
dataset = AudioMFCCDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SyllableClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(10):  # Example training for 10 epochs
    train_model(model, dataloader, criterion, optimizer)
