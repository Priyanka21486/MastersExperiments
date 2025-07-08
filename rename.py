import os
import numpy as np
import librosa
import soundfile as sf

def process_audio_files(input_folder, output_folder, target_sr=16000):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Track filenames to detect duplicates
    processed_files = set()
    unique_count = 1
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.wav', '.mp3')):
            file_path = os.path.join(input_folder, filename)
            
            # Check if file has already been processed
            if filename in processed_files:
                print(f"Duplicate filename detected, skipping: {filename}")
                continue
            
            # Load and process audio file
            y, sr = librosa.load(file_path, sr=None, mono=False)
            
            # Convert to mono if necessary
            if y.ndim > 1:
                y = np.mean(y, axis=0)
            
            # Resample to target sample rate
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            # Add filename to processed set
            processed_files.add(filename)
            
            # Create new filename
            new_filename = f"IED_{unique_count:02d}.wav"
            unique_count += 1
            new_file_path = os.path.join(output_folder, new_filename)
            
            # Save the processed audio
            sf.write(new_file_path, y, sr)
            print(f"Processed and saved: {new_filename}")

# Example usage
input_folder = '/home/spl_cair/Desktop/priyanka/icassp_exp/IED_REP_TOT'
output_folder = '/home/spl_cair/Desktop/priyanka/icassp_exp/IED_REP_OUT'
process_audio_files(input_folder, output_folder)
