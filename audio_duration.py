# Read the file and sum the minutes
def sum_minutes(file_path):
    total_minutes = 0

    with open(file_path, 'r') as file:
        for line in file:
            # Extract the number from each line
            try:
                minutes = int(line.split()[2])  # Get the first element and convert to int
                total_minutes += minutes
            except (ValueError, IndexError):
                print(f"Skipping invalid line: {line.strip()}")

    return total_minutes

# Path to your text file
file_path = '/home/spl_cair/Desktop/audio_durations.txt'
total = sum_minutes(file_path)

print(f"Total minutes: {total} mins")
