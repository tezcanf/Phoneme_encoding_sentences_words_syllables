#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEG Preprocessing Pipeline with Manual ICA Component Selection

This script preprocesses MEG data with a manual intervention point for ICA:
1. Loads and preprocesses data (filtering, resampling)
2. Fits ICA and displays components
3. STOPS for manual component selection
4. After specifying components to remove, applies ICA and saves

@author: filtsem
Created on Thu Aug 5 11:08:51 2021
"""

import os
import numpy as np
import mne
from mne.io import read_raw_ctf
from mne.preprocessing import ICA
from mne import find_events, pick_types, Annotations, write_events


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths
MEG_FOLDER = '/project/3027006.01/raw/'
OUTPUT_MEG_FOLDER = '/project/3027003.01/Filiz_folders_dont_delete/Sanne_MEG/processed'

# Subject to process
SUBJECT = 'sub-012'

# Channel type mappings
CH_TYPES = {
    'EEG057-4302': 'eog',  # EOG
    'EEG058-4302': 'eog',  # EOG
    'EEG059-4302': 'ecg',  # ECG
    'UPPT001': 'stim',     # Triggers
    'UPPT002': 'resp',     # Response
}

# Subject-specific trigger corrections
TRIGGER_DELETIONS = {
    'sub-003': [861, 1131],
    'sub-008': [646],
    'sub-009': [42, 893],
    'sub-011': [798],
    'sub-013': [917],
    'sub-018': [281, 1009, 1079],
    'sub-021': [8],
}

# Subject-specific number of trials
NUM_TRIALS = {
    'sub-018': 487,
    'sub-003': 483,
    'default': 480,
}


# ==============================================================================
# FUNCTIONS
# ==============================================================================

def load_raw_meg(subject, meg_folder):
    """
    Load raw MEG data for a subject.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-012')
    meg_folder : str
        Path to raw MEG data folder
        
    Returns
    -------
    raw : mne.io.Raw
        Raw MEG data
    raw_file_name : str
        Path to the raw file
    """
    # Determine session folder
    if subject in ['sub-011', 'sub-013']:
        session = 'ses-meg02'
    else:
        session = 'ses-meg01'
    
    meg_path = os.path.join(meg_folder, subject, session, 'meg')
    ds_data = os.listdir(meg_path)[0]
    raw_file_name = os.path.join(meg_path, ds_data)
    
    # Load and set channel types
    raw = read_raw_ctf(raw_file_name, preload=True)
    raw = raw.set_channel_types(CH_TYPES)
    
    print(f"Loaded: {raw_file_name}")
    
    return raw, raw_file_name


def clean_triggers(events, subject):
    """
    Remove extra triggers for specific subjects.
    
    Parameters
    ----------
    events : np.ndarray
        Events array from MNE
    subject : str
        Subject ID
        
    Returns
    -------
    events : np.ndarray
        Cleaned events array
    """
    if subject in TRIGGER_DELETIONS:
        indices_to_delete = TRIGGER_DELETIONS[subject]
        events = np.delete(events, indices_to_delete, axis=0)
        print(f"Removed {len(indices_to_delete)} extra triggers for {subject}")
    
    return events


def update_trigger_values(events, subject):
    """
    Update trigger values by adding 100 (stimulus) or 200 (response).
    
    Parameters
    ----------
    events : np.ndarray
        Events array
    subject : str
        Subject ID
        
    Returns
    -------
    events_new : np.ndarray
        Updated events array
    """
    events_new = events.copy()
    
    # Get number of trials for this subject
    num_trials = NUM_TRIALS.get(subject, NUM_TRIALS['default'])
    
    # Update trigger values: +100 for stim, +200 for response
    for i in range(num_trials * 2):
        if i % 2 == 0:
            events_new[i, 2] += 100  # Stimulus
        else:
            events_new[i, 2] += 200  # Response
    
    # Print trigger counts for verification
    index = np.arange(0, len(events_new) - 1, 2)
    trigger_values = events_new[index, 2]
    
    print("\nTrigger counts:")
    for trigger_id in sorted(set(trigger_values)):
        count = np.sum(trigger_values == trigger_id)
        print(f"  Trigger {trigger_id}: {count}")
    
    return events_new


def create_annotations(events_new, subject, raw):
    """
    Create annotations for bad segments (task responses and auditory localizer).
    
    Parameters
    ----------
    events_new : np.ndarray
        Updated events array
    subject : str
        Subject ID
    raw : mne.io.Raw
        Raw MEG data
        
    Returns
    -------
    annotations : mne.Annotations
        Annotations object
    """
    fs = raw.info['sfreq']
    num_trials = NUM_TRIALS.get(subject, NUM_TRIALS['default'])
    
    # Find task start times (response triggers + 1s)
    task_start = [0]
    for i in range(len(events_new)):
        if events_new[i, 2] > 200:
            task_start.append(events_new[i, 0] / fs + 1)
    
    # Find task end times (stimulus triggers - 1.5s)
    task_end = []
    for i in range(len(events_new)):
        if 100 < events_new[i, 2] < 200:
            task_end.append(events_new[i, 0] / fs - 1.5)
    
    # Handle final segment based on subject
    if subject == 'sub-020':
        # No auditory localizer after tasks
        task_end.append(events_new[num_trials * 2 - 1, 0] / fs + 1)
    else:
        task_end.append(events_new[num_trials * 2, 0] / fs)
    
    # Calculate durations
    duration_task = (np.array(task_end) - np.array(task_start)).tolist()
    
    # Annotate auditory localizer segment
    if subject == 'sub-020':
        block_empty_start = [events_new[num_trials * 2 - 1, 0] / fs + 1]
        block_empty_duration = [raw.times[-1] - block_empty_start[0]]
    else:
        block_empty_start = [events_new[num_trials * 2, 0] / fs]
        block_empty_duration = [raw.times[-1] - block_empty_start[0]]
    
    # Create annotations
    description_task = ['BAD_Task'] * len(task_start)
    description_block_empty = ['BAD_AudLoc'] * len(block_empty_start)
    
    annotations = Annotations(
        [*block_empty_start, *task_start],
        [*block_empty_duration, *duration_task],
        [*description_block_empty, *description_task]
    )
    
    return annotations


def preprocess_raw(raw, events_new, desired_sfreq=300, l_freq=1, h_freq=100):
    """
    Resample, notch filter, and bandpass filter raw data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG data
    events_new : np.ndarray
        Events array (will be updated in-place)
    desired_sfreq : int
        Target sampling frequency
    l_freq : float
        High-pass filter frequency
    h_freq : float
        Low-pass filter frequency
        
    Returns
    -------
    raw : mne.io.Raw
        Preprocessed raw data
    events_new : np.ndarray
        Updated events array
    """
    original_sfreq = raw.info['sfreq']
    
    # Resample
    print(f"Resampling from {original_sfreq} Hz to {desired_sfreq} Hz...")
    raw = raw.resample(desired_sfreq)
    
    # Update event times
    for i in range(len(events_new)):
        events_new[i, 0] = int(events_new[i, 0] * (desired_sfreq / original_sfreq))
    
    # Select MEG channels
    picks = pick_types(raw.info, meg=True, ref_meg=False, eeg=False, 
                      eog=False, stim=False)
    
    # Apply notch filter (50 Hz and harmonics)
    print("Applying notch filter at 50 and 100 Hz...")
    freqs = (50, 100)
    raw = raw.notch_filter(freqs=freqs, picks=picks)
    
    # Apply bandpass filter
    print(f"Applying bandpass filter {l_freq}-{h_freq} Hz...")
    raw = raw.filter(
        l_freq, h_freq, 
        method='fir', 
        fir_window='hamming', 
        fir_design='firwin',
        skip_by_annotation=('edge', 'BAD_Task', 'BAD_Between_bloks')
    )
    
    return raw, events_new, picks


def fit_ica(raw, picks, n_components=0.95):
    """
    Fit ICA to the data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed raw data
    picks : array
        Channel picks for ICA
    n_components : float
        Number of ICA components (fraction of explained variance)
        
    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    """
    print("\nFitting ICA...")
    ica = ICA(method='fastica', random_state=45, max_iter=10000, 
              n_components=n_components)
    ica.fit(raw, picks=picks, decim=None)
    print("ICA fitted successfully")
    
    return ica


def apply_ica(raw, ica, removed_components):
    """
    Apply ICA after manual component selection.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed raw data
    ica : mne.preprocessing.ICA
        Fitted ICA object
    removed_components : list
        List of component indices to remove
        
    Returns
    -------
    raw : mne.io.Raw
        Cleaned raw data
    """
    print(f"\nRemoving components: {removed_components}")
    ica.exclude.extend(removed_components)
    raw = ica.apply(raw)
    raw = raw.set_channel_types(CH_TYPES)
    
    return raw


def save_preprocessed_data(raw, events_new, subject, output_folder):
    """
    Save preprocessed raw data and events.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed raw data
    events_new : np.ndarray
        Events array
    subject : str
        Subject ID
    output_folder : str
        Path to output folder
    """
    # Create output directory if needed
    subject_output = os.path.join(output_folder, subject, 'meg')
    os.makedirs(subject_output, exist_ok=True)
    
    # Save events
    event_file_name = os.path.join(subject_output, f'{subject}-eve.fif')
    write_events(event_file_name, events_new)
    print(f"Saved events to: {event_file_name}")
    
    # Save raw data
    raw_file_name = os.path.join(subject_output, f'{subject}_resampled_300Hz-ICA-raw.fif')
    raw.save(raw_file_name, overwrite=True)
    print(f"Saved preprocessed data to: {raw_file_name}")


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

print(f"\n{'='*70}")
print(f"MEG PREPROCESSING - SUBJECT: {SUBJECT}")
print(f"{'='*70}\n")

# Step 1: Load raw data
print("STEP 1: Loading raw MEG data...")
raw, raw_file_name = load_raw_meg(SUBJECT, MEG_FOLDER)
fs = raw.info['sfreq']

# Step 2: Find and clean events
print("\nSTEP 2: Processing events...")
events = find_events(raw, shortest_event=1)
print(f"Found {len(events)} events")
events = clean_triggers(events, SUBJECT)

# Step 3: Update trigger values
print("\nSTEP 3: Updating trigger values...")
events_new = update_trigger_values(events, SUBJECT)

# Step 4: Create annotations
print("\nSTEP 4: Creating annotations for bad segments...")
annotations = create_annotations(events_new, SUBJECT, raw)
raw = raw.set_annotations(annotations)
print(f"Added {len(annotations)} annotations")

# Step 5: Preprocess (resample, filter)
print("\nSTEP 5: Preprocessing (resample, notch, bandpass)...")
events_new_org = events_new.copy()  # Keep original for reference
raw, events_new, picks = preprocess_raw(raw, events_new)

# Step 6: Fit ICA
print("\nSTEP 6: Fitting ICA...")
ica = fit_ica(raw, picks)

# Step 7: Display ICA components for manual inspection
print("\nSTEP 7: Displaying ICA components for manual inspection...")
print("-" * 70)
print("MANUAL INTERVENTION REQUIRED:")
print("  1. Inspect the ICA components in the plots")
print("  2. Identify components to remove (artifacts)")
print("  3. Add component numbers to 'removed_components' list below")
print("  4. Re-run from the 'Apply ICA' section")
print("-" * 70)

ica.plot_components()
ica.plot_sources(raw, show_scrollbars=False)

# STOP HERE - User needs to inspect components
print("\n" + "="*70)
print("STOPPING FOR MANUAL COMPONENT SELECTION")
print("="*70)
print("\nAfter inspecting the plots:")
print("1. Close the plot windows")
print("2. Edit the 'removed_components' list below")
print("3. Comment out the line 'raise SystemExit(...)'")
print("4. Re-run the script from this point")
print("\n" + "="*70 + "\n")

raise SystemExit("Please inspect ICA components and specify which to remove.")

# ==============================================================================
# MANUAL COMPONENT SELECTION
# ==============================================================================
# After inspecting the ICA plots, specify the component numbers to remove:
removed_components = []  # e.g., [0, 5, 12] - ADD COMPONENT NUMBERS HERE

# ==============================================================================
# APPLY ICA AND SAVE
# ==============================================================================

print(f"\n{'='*70}")
print(f"CONTINUING WITH COMPONENT REMOVAL")
print(f"{'='*70}\n")

# Step 8: Apply ICA with selected components
print("STEP 8: Applying ICA...")
raw = apply_ica(raw, ica, removed_components)

# Step 9: Plot to verify
print("\nSTEP 9: Plotting for visual verification...")
events = find_events(raw, shortest_event=1)
raw.plot(events_new)

# Step 10: Save preprocessed data
print("\nSTEP 10: Saving preprocessed data...")
save_preprocessed_data(raw, events_new, SUBJECT, OUTPUT_MEG_FOLDER)

print(f"\n{'='*70}")
print(f"PREPROCESSING COMPLETE FOR {SUBJECT}")
print(f"{'='*70}\n")