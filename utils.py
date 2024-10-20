import re
from datetime import datetime
from pathlib import Path

def parse_time(time_str):
    """Parse time string in the format HH.MM.SS into a datetime object."""
    return datetime.strptime(time_str, '%H.%M.%S')

def calculate_time_difference(start_time, event_time):
    """Calculate the time difference in seconds between two datetime objects."""
    delta = event_time - start_time
    return delta.total_seconds()

def read_seizure_times(filepath):
    seizure_times = []
    
    # Regular expressions to capture registration and seizure times
    registration_start_pattern = re.compile(r"Registration start time: (\d{2}\.\d{2}\.\d{2})")
    registration_end_pattern = re.compile(r"Registration end time: (\d{2}\.\d{2}\.\d{2})")
    seizure_start_pattern = re.compile(r"Seizure start time: (\d{2}\.\d{2}\.\d{2})")
    seizure_end_pattern = re.compile(r"Seizure end time: (\d{2}\.\d{2}\.\d{2})")

    # Use Path to handle the file path
    file_path = Path(filepath)
    
    with file_path.open('r') as file:
        registration_start = None
        registration_end = None
        seizure_start = None
        seizure_end = None
        
        for line in file:
            reg_start_match = registration_start_pattern.search(line)
            reg_end_match = registration_end_pattern.search(line)
            seizure_start_match = seizure_start_pattern.search(line)
            seizure_end_match = seizure_end_pattern.search(line)

            if reg_start_match:
                registration_start = parse_time(reg_start_match.group(1))
            if reg_end_match:
                registration_end = parse_time(reg_end_match.group(1))
            if seizure_start_match:
                seizure_start = parse_time(seizure_start_match.group(1))
            if seizure_end_match:
                seizure_end = parse_time(seizure_end_match.group(1))

                if registration_start and seizure_start and seizure_end:
                    # Calculate the start and end times relative to registration start
                    relative_seizure_start = calculate_time_difference(registration_start, seizure_start)
                    relative_seizure_end = calculate_time_difference(registration_start, seizure_end)
                    
                    # Store the results as a tuple (start, end) in the seizure_times array
                    seizure_times.append((relative_seizure_start, relative_seizure_end))
    
    return seizure_times