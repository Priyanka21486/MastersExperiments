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

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Conv1D, Flatten, Masking
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from scipy.stats import skew


# import librosa
# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

# Define the BiLSTM model
def create_bilstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=input_shape)),
        Bidirectional(LSTM(64)),
        Dense(39, activation='linear')  # Output dimension is 13
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Example input shape for the model
input_shape = (None, 39)  # None for variable time steps, 13 for MFCC features
model = create_bilstm_model(input_shape)
def extract_mfcc_features(audio, sr, syllable_boundaries, model):
    mfcc_features = []
    
    for start, end in syllable_boundaries:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        syllable_audio = audio[start_sample:end_sample]
        mfcc = librosa.feature.mfcc(y=syllable_audio, sr=sr, n_fft=1024, n_mfcc=39)
        
        # Ensure MFCC has the right shape
        mfcc = np.expand_dims(mfcc.T, axis=0)  # Shape (1, time_steps, 13)
        
        # Use the BiLSTM model to extract features
        syllable_features = model.predict(mfcc)
        
        # Append the features (1, 13)
        mfcc_features.append(syllable_features.squeeze())  # Shape (13,)
    # print(mfcc_features)
    return np.array(mfcc_features)  # Shape (no_of_syllables, 13)

def extract_prosody_features(audio, sr, syllable_boundaries):
    prosody_features = []
    for start, end in syllable_boundaries:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        syllable_audio = audio[start_sample:end_sample]

        # F0 contour and its statistics
        f0, _ = librosa.piptrack(y=syllable_audio, sr=sr)
        f0 = np.mean(f0, axis=0)
        f0 = f0[f0 > 0]  # Remove zeros
        if len(f0) == 0:
            f0 = np.array([0])
        f0_mean = np.mean(f0)
        f0_std = np.std(f0)
        f0_skew = skew(f0)
        f0_mse = np.mean((f0 - f0_mean) ** 2)
        if len(f0) > 1:
            try:
                f0_tilt = np.polyfit(np.arange(len(f0)), f0, 1)[0]  # Slope as tilt
            except np.linalg.LinAlgError:
                f0_tilt = 0
        else:
            f0_tilt = 0
        # f0_tilt = np.polyfit(np.arange(len(f0)), f0, 1)[0]  
        
        f0_tilt_mean=np.mean(f0_tilt)
        f0_tilt_std=np.std(f0_tilt)
        f0_tilt_skew=skew(f0_tilt)# Slope as tilt

        # Energy contour and its statistics
        energy = librosa.feature.rms(y=syllable_audio)[0]
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        # energy_skew = skew(energy)
        energy_mse = np.mean((energy - energy_mean) ** 2)
        # energy_tilt = np.polyfit(np.arange(len(energy)), energy, 1)[0]  # Slope as tilt

        # Duration features
        duration = end - start
        voiced_duration = np.sum(librosa.effects.split(syllable_audio, top_db=20)) / sr
        unvoiced_duration = duration - voiced_duration
        pause_duration = duration - voiced_duration
        voiced_to_unvoiced_ratio = voiced_duration / (unvoiced_duration + 1e-6)
        voiced_to_pause_ratio = voiced_duration / (pause_duration + 1e-6)
        unvoiced_to_pause_ratio = unvoiced_duration / (pause_duration + 1e-6)
        
        pause_durations = np.array(librosa.effects.split(syllable_audio, top_db=20)) / sr
        pause_mean = np.mean(pause_durations)
        pause_std = np.std(pause_durations)
        # pause_min = np.min(pause_durations)
        # pause_max = np.max(pause_durations)
        # pause_skew = skew(pause_durations)
        # print(pause_skew)
        
        # Append all features
        features = [
            f0_mean, f0_std, f0_skew, f0_mse, f0_tilt_mean,f0_tilt_std, f0_tilt_skew,
            energy_mean, energy_std, energy_mse,
            voiced_to_unvoiced_ratio, voiced_to_pause_ratio, unvoiced_to_pause_ratio,
            pause_mean, pause_std
        ]
        # prosody_features.append(features)
        features=np.array(features)
        # print(f"prosody----------{features}")

        # if features.shape[0] != 19:  # Assuming 18 features
        #     raise ValueError(f"Unexpected feature length: {features.shape[0]}")

        
        prosody_features.append(features.squeeze())
        # print(f"prosody feats = {prosody_features}")
    
    return np.array(prosody_features)  # Shape (number_of_syllables, number_of_features)

def combine_mfcc_and_prosody(audio, sr, syllable_boundaries):
    mfcc_features = extract_mfcc_features(audio, sr, syllable_boundaries,model)

    prosody_features = extract_prosody_features(audio, sr, syllable_boundaries)
    print(len(mfcc_features))
    print(len(prosody_features))

    # Ensure features align
    if mfcc_features.shape[0] != prosody_features.shape[0]:
        raise ValueError("Mismatch between number of syllables in MFCC and prosody features")

    # Concatenate MFCC and prosody features
    combined_features = np.hstack([mfcc_features, prosody_features])
    return combined_features  # Shape (number_of_syllables, 31)

def syllable_boundaries_function(file_path):
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
    wav_data, fs = librosa.load(file_path)
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

def load_data_from_folders(folder_paths, syllable_boundaries_function):
    X = []
    y = []
    labels = {folder: idx for idx, folder in enumerate(folder_paths)}

    for folder in tqdm(folder_paths):
        for filename in os.listdir(folder):
            if filename.endswith (".wav"):
                file_path = os.path.join(folder, filename)
                audio, sr = librosa.load(file_path, sr=None) 
                syllable_boundaries = syllable_boundaries_function(file_path)  # Define this function
                # mfcc_features = extract_mfcc_from_syllables(file_path, syllable_boundaries)
                # mfcc_features = extract_mfcc_features(audio, sr, syllable_boundaries, model)
                combine_features=combine_mfcc_and_prosody(audio, sr, syllable_boundaries)
                combine_features = np.array(combine_features)
                print(f"combine------------{combine_features}{len(combine_features[0])}")

                if len(combine_features) < 20:
            # Pad with zeros
                  padding = np.zeros((20 - len(combine_features), 54))
                  combine_features = np.vstack([combine_features, padding])
                  print(combine_features.shape) 
                elif len(combine_features) > 20:
            # Truncate
                  combine_features = combine_features[:20]
                  print(combine_features.shape)  
                X.append((combine_features))
                y.append([labels[folder]])


    return X, y
folder_paths = ['/home/spl_cair/Desktop/priyanka/icassp_exp/TISA_REP_OUT_PART', '/home/spl_cair/Desktop/priyanka/icassp_exp/IED_REP_OUT_PART']
X, y = load_data_from_folders(folder_paths, syllable_boundaries_function)
X=np.array(X)
y=np.array(y)
print(y.shape)
print(y)
# import numpy as np
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)
import tensorflow as tf
from tensorflow.keras.layers import Input, Masking, TimeDistributed, Conv1D, Flatten, Dense
from tensorflow.keras.models import Model

# Define the model
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Masking layer to handle padded values
    x = Masking(mask_value=0.0)(inputs)
    
    # TimeDistributed 1D Convolutional layer
    x = (Conv1D(filters=64, kernel_size=5, dilation_rate=1 ,activation='relu', padding='same'))(x)
    x = (Conv1D(filters=128, kernel_size=3, dilation_rate=2 ,activation='relu', padding='same'))(x)
    
    # Flatten the output
    x = (Flatten())(x)
    
    # Fully connected layers
    x = (Dense(64, activation='relu'))(x)
    x = (Dense(32, activation='relu'))(x)

    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)

    return model

model=build_model((20,54))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
# Evaluate the model

evaluation_result = model.evaluate(X_test, y_test)

# Print the result to understand its structure
print(f"Evaluation result: {evaluation_result}")

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

#y_pred = np.argmax(y_pred_prob, axis=1)
print(y_pred)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
