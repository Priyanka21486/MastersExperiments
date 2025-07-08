import os
import numpy as np
from pydub import AudioSegment

def parse_annotation_file(file_path):
    annotations = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                start_time, end_time, label = parts
                annotations.append({
                    'start_time': float(start_time),
                    'end_time': float(end_time),
                    'label': label
                })
    return annotations

def extract_audio_segment(audio, start_time, duration_ms):
    start_ms = start_time # Convert seconds to milliseconds
    end_ms = start_ms + duration_ms
    print(start_ms,end_ms)
    return audio[start_ms:end_ms]

def save_segment(segment, output_path):
    segment.export(output_path, format='wav')

def process_audio_file(audio_file_path, annotations, output_folder):
    audio = AudioSegment.from_wav(audio_file_path)
    print(f"len is {len(audio)}")
    for annotation in annotations:
        start_time = annotation['start_time']
        end_time = annotation['end_time']
        label = annotation['label']
        print(start_time)
        # Determine the start time of the 3-second segment to include the annotation
        segment_start_time = max(0, start_time - 1.5)  # 1.5 seconds before the start of the annotation
        segment_end_time = segment_start_time + 3 # 3 seconds later (3000 ms)
        print(segment_start_time,segment_end_time)
        
        if segment_end_time > len(audio):
            segment_end_time = len(audio)
            segment_start_time = segment_end_time - 3
        
        segment = extract_audio_segment(audio, segment_start_time * 1000, 3000)
        
        # Create label directory if it doesn't exist
        label_folder = os.path.join(output_folder, label)
        os.makedirs(label_folder, exist_ok=True)
        
        # Generate a unique filename
        segment_filename = f"segment_{start_time:.2f}_{end_time:.2f}.wav"
        segment_path = os.path.join(label_folder, segment_filename)
        
        save_segment(segment, segment_path)

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            txt_file_path = os.path.join(folder_path, filename)
            audio_file_path = os.path.join(folder_path, filename.replace('.txt', '.wav'))
            
            if os.path.exists(audio_file_path):
                annotations = parse_annotation_file(txt_file_path)
                output_folder = os.path.join(folder_path, 'output')
                process_audio_file(audio_file_path, annotations, output_folder)

folder_path = '/home/spl_cair/Desktop/priyanka/intern4/aud+txt'
process_folder(folder_path)

