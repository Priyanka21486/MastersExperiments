import os
import csv
import pandas as pd
from pytube import YouTube

# Function to download a YouTube video and return its duration
def download_video(link, folder, filename):
    try:
        youtube_object = YouTube(link)
        stream = youtube_object.streams.get_highest_resolution()
        stream.download(output_path=folder, filename=filename)
        print(f"Downloaded: {filename} to {folder}")
        return youtube_object.length  # Return video duration in seconds
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

# Read the input data
input_data = csv.DictReader(open('Stutter_Video_Dataset.csv'))

# Dictionary to keep track of the number of videos downloaded per speaker and task
task_speaker_count = {}
# Dictionary to keep track of the number of videos and total duration per speaker
speaker_stats = {}

# Download each video
for entry in input_data:
    task = entry["Task"].replace("Mod ", "").replace(".", "_")
    speaker = entry["Speaker"]
    link = entry["Link"]
    folder = f"Speaker_{speaker}"
    
    # Create directory if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Ensure unique filenames by counting the number of videos for each task and speaker
    if (task, speaker) in task_speaker_count:
        task_speaker_count[(task, speaker)] += 1
    else:
        task_speaker_count[(task, speaker)] = 1
    video_number = task_speaker_count[(task, speaker)]
    filename = f"{task}_Speaker_{speaker}_{video_number:03d}.mp4"
    
    # Download the video and get its duration
    duration = download_video(link, folder, filename)
    
    # Update speaker stats
    if speaker in speaker_stats:
        speaker_stats[speaker]['Total_Videos_Downloaded'] += 1
        speaker_stats[speaker]['Total_Duration'] += duration
    else:
        speaker_stats[speaker] = {'Total_Videos_Downloaded': 1, 'Total_Duration': duration}

# Write the CSV file
csv_filename = "video_downloads_summary.csv"
with open(csv_filename, mode='w', newline='') as csv_file:
    fieldnames = ['Speaker', 'Total_Videos_Downloaded', 'Total_Duration']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for speaker, stats in speaker_stats.items():
        writer.writerow({
            'Speaker': speaker,
            'Total_Videos_Downloaded': stats['Total_Videos_Downloaded'],
            'Total_Duration': stats['Total_Duration']
        })

print(f"Summary CSV created: {csv_filename}")