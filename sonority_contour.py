import numpy as np
import librosa
import matplotlib.pyplot as plt

def theta_oscillator(t, frequency, amplitude):
    """Generate a theta oscillator signal."""
    return amplitude * np.sin(2 * np.pi * frequency * t)

def get_sonority_contour_with_theta_and_boundaries(audio_path, syllable_boundaries=None, sr=22050, theta_freq=5.0, theta_amp=1.0):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)

    # Compute the fundamental frequency (F0) using librosa's pyin function
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

    # Replace NaN values with zero or interpolate
    f0 = np.nan_to_num(f0, nan=0.0)

    # Compute time axis for F0
    times = librosa.times_like(f0, sr=sr)

    # Create a theta oscillator signal
    theta_signal = theta_oscillator(times, theta_freq, theta_amp)

    # Normalize F0 and theta signal for comparison
    f0_norm = (f0 - np.min(f0)) / (np.max(f0) - np.min(f0))
    theta_signal_norm = (theta_signal - np.min(theta_signal)) / (np.max(theta_signal) - np.min(theta_signal))

    # Create time axis for the raw audio signal
    y_times = np.arange(len(y)) / sr

    # Plot the figures
    plt.figure(figsize=(14, 10))

    # Plot the raw audio signal
    plt.subplot(3, 1, 1)
    plt.plot(y_times, y, label='Raw Audio Signal', color='gray')
    if syllable_boundaries:
        for boundary in syllable_boundaries:
            plt.axvline(x=boundary, color='red', linestyle='--', label='Syllable Boundary')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Raw Audio Signal')
    plt.grid()
    plt.legend()

    # Plot the sonority contour (F0) overlapping with the audio signal
    plt.subplot(3, 1, 2)
    plt.plot(y_times[:len(f0)], f0_norm, label='Sonority Contour (Normalized F0)', color='blue')
    if syllable_boundaries:
        for boundary in syllable_boundaries:
            plt.axvline(x=boundary, color='red', linestyle='--', label='Syllable Boundary')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized F0')
    plt.title('Sonority Contour')
    plt.grid()
    plt.legend()

    # Plot the theta oscillator signal
    plt.subplot(3, 1, 3)
    plt.plot(times, theta_signal_norm, label='Theta Oscillator Signal (Normalized)', color='red', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.title('Theta Oscillator Signal')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

    return f0, times, theta_signal, y, y_times

# Example usage
audio_path = '/home/spl_cair/Desktop/priyanka/Tisa_DysTypes/PWR/pwr.wav'
# Example syllable boundaries (in seconds)
syllable_boundaries = [0.5, 1.2, 2.8]  # Replace these with actual boundary times if available
f0, times, theta_signal, y, y_times = get_sonority_contour_with_theta_and_boundaries(audio_path, syllable_boundaries)


