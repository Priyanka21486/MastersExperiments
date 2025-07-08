import os
import librosa

def check_audio_channels(audio_path):
    """
    Check if the audio file is mono or stereo.
    """
    # Load audio file with librosa
    y, sr = librosa.load(audio_path, sr=None)
    
    # Check the number of channels
    if len(y.shape) == 1:
        return 'Mono'
    elif len(y.shape) == 2:
        return 'Stereo'
    else:
        return 'Unknown'

def check_audio_folder(folder_path):
    """
    Check all audio files in a folder and determine if they are mono or stereo.
    """
    audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]  # Adjust extension if needed
    results = {}
    
    for audio_file in audio_files:
        audio_path = os.path.join(folder_path, audio_file)
        channel_type = check_audio_channels(audio_path)
        if channel_type == 'Mono':
            results[audio_file] = channel_type
    
    return results

# Define the path to your audio folder
folder_path = '/home/spl_cair/Desktop/priyanka/icassp_exp/TISA_PR'

# Check all audio files in the folder
results = check_audio_folder(folder_path)

# Print results for stereo files only
print("Stereo Audio Files:")
for file_name, channel_type in results.items():
    print(f"{file_name}: {channel_type}")
