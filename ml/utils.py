import re
from datetime import datetime, timedelta
from pathlib import Path
import collections
import mne
import numpy as np

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


def createCroppedFif(datasets, epilepsy):
    cropped_ep_edf_map = {}  # Dictionary to store the mapping of cropped EDF files to subjects
    output_folder = Path(datasets, "cropped_ep_dataset") # Folder to save the cropped EDF files
    output_folder.mkdir(exist_ok=True)  # Create the output folder if it doesn't exist

    i = 1
    for pn_folder in epilepsy.glob("PN*"):
        if pn_folder.is_dir():
            print(f"processing folder: {pn_folder.name}")

            subj = Path(epilepsy, pn_folder.name)
            seizure_list_path = Path(subj, f"Seizures-list-{pn_folder.name}.txt")
            seizure_times_map = read_seizure_times(seizure_list_path)
            print(seizure_times_map)
            # iterate through .edf files
            for idx, edf_file in enumerate(pn_folder.glob("*.edf")):
                print(f"  Found EDF file: {edf_file.name}")

                start_end_pairs_list = seizure_times_map.get(edf_file.name, [])

                for start, end in start_end_pairs_list:
                    epileform_start = start - 360
                    epileform_end = start - 60

                    print(f"    File: {edf_file.name} -> start: {epileform_start} | end: {epileform_end}")

                    ep_edf = mne.io.read_raw_edf(
                        Path(subj, edf_file.name),
                        preload=True,
                        infer_types=True,
                        # exclude=["EKG EKG", "SPO2", "HR", "1", "2", "MK"],
                        # exclude=["SPO2", "HR", "1", "2", "MK"],
                        include=['Fp1', 'Fp2', 'Cp5', 'Cp6', 'C3', 'C4', 'O1', 'O2'],
                        verbose=False,
                    )

                    # ep_edf = ep_edf.filter(l_freq=1, h_freq=40, verbose=False).notch_filter(60, verbose=False)
                    ep_edf.set_montage("standard_1020", on_missing="ignore")
                    ep_edf = ep_edf.crop(tmin=epileform_start, tmax=epileform_end)

                    cropped_filename = f"cropped_ep{i}_raw.fif"
                    output_path = output_folder / cropped_filename
                    
                    ep_edf.save(output_path, overwrite=True, verbose=False)
                    cropped_ep_edf_map[cropped_filename] = subj.name

                    i += 1
                    # print(ep_edf.times)
    return cropped_ep_edf_map

def createEpochFif(cropped_ep_edf_map):
    epoch_subj_map = {}
    cropped_dataset_path = Path("Datasets/cropped_ep_dataset")
    i = 1
    for cropped_ep in cropped_dataset_path.glob("*"):
        print(cropped_ep.name)

        raw = mne.io.read_raw_fif(Path(cropped_dataset_path, cropped_ep.name), preload=True)

        raw.filter(l_freq=1, h_freq=40, verbose=False).notch_filter(60, verbose=False)
        raw.resample(200)

        epochs = mne.make_fixed_length_epochs(raw, duration=15, preload=False)

        epoch_filename = f"epoch{i}-raw.fif"
        epochs.save(Path("Datasets/epoch_ep_dataset", epoch_filename), overwrite=True, verbose=False)

        epoch_subj_map[epoch_filename] = cropped_ep_edf_map.get(cropped_ep.name)
        i += 1

    return epoch_subj_map

def createHealthyFif(healthy):
    base_path = healthy
    for sub_folder in base_path.glob("sub-*"):
        ses1_folder = sub_folder / 'ses-1'

        print(sub_folder.name)
        if ses1_folder.exists():
            eeg_folder = ses1_folder / 'eeg'

            print(f"  {ses1_folder.name}")
            if eeg_folder.exists():
                # print(f"    {sub_folder.name}_ses-1_task-eyeclosed_eeg.set")
                fdt_file = eeg_folder.glob(f"{sub_folder.name}_ses-1_task-eyesclosed_eeg.set")

                for file in fdt_file:
                    print(f"    {file.name}")
    

