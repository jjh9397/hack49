import re
from datetime import datetime, timedelta
from pathlib import Path
import collections

def parse_time(time_str):
    """Parse time string in the format HH.MM.SS into a datetime object."""
    return datetime.strptime(time_str, '%H.%M.%S')

def calculate_time_difference(start_time, event_time):
    """Calculate the time difference in seconds between two datetime objects."""
    delta = event_time - start_time
    return delta.total_seconds()

def handle_wrap_around(start_time, event_time):
    """Handle the case where event_time is 'earlier' than start_time due to 24-hour wrap-around."""
    if event_time < start_time:
        # Add 24 hours to event_time to account for wrap-around past midnight
        event_time += timedelta(days=1)
    return event_time

def read_seizure_times(filepath):
    file_map = collections.defaultdict(list)
    seizure_times = []
    
    # Regular expressions to capture registration and seizure times
    file_name_pattern = re.compile(r"File name:\s*(\S+\.edf)")
    registration_start_pattern = re.compile(r"Registration start time:\s*(\d{2}\.\d{2}\.\d{2})")
    registration_end_pattern = re.compile(r"Registration end time:\s*(\d{2}\.\d{2}\.\d{2})")
    seizure_start_pattern = re.compile(r"Seizure start time:\s*(\d{2}\.\d{2}\.\d{2})")
    seizure_end_pattern = re.compile(r"Seizure end time:\s*(\d{2}\.\d{2}\.\d{2})")

    # Use Path to handle the file path
    file_path = Path(filepath)
    
    # Lists to store all times
    file_names = []
    registration_starts = []
    registration_ends = []
    seizure_starts = []
    seizure_ends = []
    
    # Read and parse the file
    with file_path.open('r') as file:
        for line in file:
            file_match = file_name_pattern.search(line)
            reg_start_match = registration_start_pattern.search(line)
            reg_end_match = registration_end_pattern.search(line)
            seizure_start_match = seizure_start_pattern.search(line)
            seizure_end_match = seizure_end_pattern.search(line)

            if file_match:
                file_names.append(file_match.group(1))

            if reg_start_match:
                registration_starts.append(parse_time(reg_start_match.group(1)))

            if reg_end_match:
                registration_ends.append(parse_time(reg_end_match.group(1)))

            if seizure_start_match:
                seizure_starts.append(parse_time(seizure_start_match.group(1)))

            if seizure_end_match:
                seizure_ends.append(parse_time(seizure_end_match.group(1)))

    # Log the parsed data to check the lists
    # print(f"Registration Starts: {registration_starts}")
    # print(f"Registration Ends: {registration_ends}")
    # print(f"Seizure Starts: {seizure_starts}")
    # print(f"Seizure Ends: {seizure_ends}")
    
    # Check that all lists have matching lengths (number of events is the same)
    if not (len(registration_starts) == len(registration_ends) == len(seizure_starts) == len(seizure_ends)):
        raise ValueError("Mismatched number of registration or seizure start/end times")

    # Process times by index
    for i in range(len(file_names)):
        reg_start = registration_starts[i]
        reg_end = registration_ends[i]
        seizure_start = seizure_starts[i]
        seizure_end = seizure_ends[i]

        # Handle wrap-around if the seizure start/end time is earlier than the registration start
        seizure_start = handle_wrap_around(reg_start, seizure_start)
        seizure_end = handle_wrap_around(reg_start, seizure_end)

        # Calculate the start and end times relative to registration start
        relative_seizure_start = calculate_time_difference(reg_start, seizure_start)
        relative_seizure_end = calculate_time_difference(reg_start, seizure_end)

        # Append results as a tuple (start, end) to the seizure_times list
        file_map[file_names[i]].append((relative_seizure_start, relative_seizure_end))

    return file_map


import mne
import numpy as np
from pathlib import Path

def create_15_sec_epochs(edf_list):
    epochs_dict = {}

    # Iterate over each EDF file
    for edf_file in edf_list:
        # Load the EDF file using MNE
        raw = mne.io.read_raw_edf(edf_file, preload=True)

        # Get the total duration of the recording in seconds (5 minutes = 300 seconds)
        total_duration = raw.times[-1]

        # Ensure that the file is indeed a 5-minute file (300 seconds)
        if total_duration != 300:
            raise ValueError(f"EDF file {edf_file} is not 5 minutes long (duration: {total_duration} seconds).")

        # Calculate the number of 15-second chunks
        n_chunks = int(total_duration / 15)  # 300 seconds divided into 15-second intervals

        # Create artificial events every 15 seconds
        events = np.array([[int(i * raw.info['sfreq'] * 15), 0, 1] for i in range(n_chunks)])

        # Define event_id for the epochs
        event_id = {'15_sec_chunk': 1}

        # Create epochs with a duration of 15 seconds
        epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=15, baseline=None, preload=True)

        # Store the epochs in a dictionary with the EDF file name as the key
        epochs_dict[edf_file] = epochs

    return epochs_dict
