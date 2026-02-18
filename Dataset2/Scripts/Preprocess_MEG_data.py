#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEG Preprocessing Pipeline with ICA
- Loads raw CTF MEG data
- Removes incomplete trials
- Annotates task/response periods as BAD
- Resamples and filters
- Applies ICA for artifact removal
- Saves ICA-cleaned data

@author: filiztezcan
"""

import os
import numpy as np
from pathlib import Path
from mne.io import read_raw_ctf
from mne import find_events, pick_types, Annotations, write_events
from mne.preprocessing import ICA

# =============================================================================
# CONFIGURATION
# =============================================================================

# Choose dataset: 'Dutch_participants' or 'Chinese_participants'
DATASET = 'Dutch_participants'

# Paths
root = Path.cwd().parents[1]
MEG_folder = root / DATASET / 'raw'
Output_MEG_folder = root / DATASET / 'processed'

# Subject info
subject = 'sub-003'
STIMULUS_TYPE = 'Dutch_stimuli'  # or 'Chinese_stimuli'

# Channel type mapping
ch_types = {
    'EEG057-4302': 'eog',      # EOG
    'EEG058-4302': 'eog',      # EOG
    'EEG059-4302': 'ecg',      # EKG
    'UPPT001': 'stim',         # Triggers
    'UPPT002': 'resp',         # Response
}

# Processing parameters
DESIRED_SFREQ = 300  # Hz
L_FREQ = 1           # High-pass filter (Hz)
H_FREQ = 100         # Low-pass filter (Hz)
NOTCH_FREQS = (50, 100)

# ICA parameters
ICA_METHOD = 'fastica'
ICA_RANDOM_STATE = 42
ICA_MAX_ITER = 10000
N_COMPONENTS = 40


# =============================================================================
# LOAD RAW DATA
# =============================================================================

def load_raw_data(subject, MEG_folder, ch_types):
    """Load raw MEG data and set channel types."""
    ds_data = os.listdir(os.path.join(MEG_folder, subject, 'ses-meg02/meg'))[0]
    raw_file_name = os.path.join(MEG_folder, subject, 'ses-meg02/meg/', ds_data)
    
    print(f"Loading: {raw_file_name}")
    
    raw = read_raw_ctf(raw_file_name, preload=True)
    raw = raw.set_channel_types(ch_types)
    events = find_events(raw, shortest_event=1)
    
    fs = raw.info['sfreq']
    print(f"Sampling frequency: {fs} Hz")
    
    return raw, events, fs


# =============================================================================
# CLEAN EVENTS
# =============================================================================

def remove_triggers(events):
    """Remove triggers with codes 55 or 77."""
    to_be_deleted = []
    
    for i in range(len(events)):
        if events[i, 2] == 55 or events[i, 2] == 77:
            # Find previous trial start marker
            j = i
            while events[j, 2] not in [10, 20, 30]:
                j = j - 1
            to_be_deleted.extend([j, j+1])
    
    events_clean = np.delete(events, to_be_deleted, 0)
    
    # Print trial counts
    print(f"Trials with code 10: {len(np.where(events_clean == 10)[0])}")
    print(f"Trials with code 20: {len(np.where(events_clean == 20)[0])}")
    print(f"Trials with code 30: {len(np.where(events_clean == 30)[0])}")
    
    return events_clean


# =============================================================================
# ANNOTATE BAD SEGMENTS
# =============================================================================

def annotate_task_periods(raw, events, fs):
    """Annotate task response and between-block periods as BAD."""
    # Find task start times (2 seconds after trial start)
    task_start = [0]
    for i in range(len(events)):
        if events[i, 2] in [11, 21, 31]:
            task_start.append(events[i, 0] / fs + 2)
    
    # Find task end times (1 second before trial end)
    task_end = []
    for i in range(len(events)):
        if events[i, 2] in [10, 20, 30]:
            task_end.append(events[i, 0] / fs - 1)
    task_end.append(raw.last_samp / fs)
    
    # Calculate durations
    duration = np.array(task_end) - np.array(task_start)
    description = ['BAD_Task'] * len(task_start)
    
    # Create annotations
    annotations = Annotations(task_start, duration.tolist(), description)
    raw = raw.set_annotations(annotations)
    
    return raw


# =============================================================================
# FILTERING AND RESAMPLING
# =============================================================================

def resample_and_filter(raw, events, fs, desired_sfreq, l_freq, h_freq, notch_freqs):
    """Resample and apply notch and bandpass filters."""
    # Update events for new sampling rate
    events_resampled = events.copy()
    
    # Resample
    raw = raw.resample(desired_sfreq)
    
    # Update event timings
    for i in range(len(events_resampled)):
        events_resampled[i, 0] = int(events_resampled[i, 0] * (desired_sfreq / fs))
    
    # Apply notch filter
    picks = pick_types(raw.info, meg=True, ref_meg=False, eeg=False, 
                      eog=False, stim=False)
    raw = raw.notch_filter(freqs=notch_freqs, picks=picks)
    
    # Apply bandpass filter
    raw = raw.filter(l_freq, h_freq, method='fir', fir_window='hamming',
                     fir_design='firwin', 
                     skip_by_annotation=('edge', 'BAD_Task', 'BAD_Between_bloks'))
    
    return raw, events_resampled, picks


# =============================================================================
# ICA PROCESSING
# =============================================================================

def fit_ica(raw, picks, method='fastica', n_components=40, random_state=42, max_iter=10000):
    """
    Fit ICA to raw MEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG data
    picks : list
        MEG channel picks
    method : str
        ICA method (default: 'fastica')
    n_components : int
        Number of ICA components (default: 40)
    random_state : int
        Random state for reproducibility
    max_iter : int
        Maximum number of iterations
    
    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    """
    print("\n" + "=" * 70)
    print("FITTING ICA")
    print("=" * 70)
    print(f"Method: {method}")
    print(f"Components: {n_components}")
    print(f"Random state: {random_state}")
    
    ica = ICA(
        method=method,
        random_state=random_state,
        max_iter=max_iter,
        n_components=n_components
    )
    
    # Fit ICA on MEG channels only
    ica.fit(raw, picks=picks, decim=None)
    print("ICA fitted successfully")
    
    return ica


def visualize_ica_components(ica, raw):
    """
    Plot ICA components and sources for visual inspection.
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.io.Raw
        Raw MEG data
    """
    print("\nVisualizing ICA components...")
    
    # Plot component topographies
    ica.plot_components()
    
    # Plot component time courses
    ica.plot_sources(raw, show_scrollbars=False)


def select_artifact_components(ica, component_indices):
    """
    Manually select ICA components to exclude.
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    component_indices : list
        List of component indices to exclude (e.g., [12, 13])
    
    Returns
    -------
    ica : mne.preprocessing.ICA
        ICA object with excluded components
    """
    print(f"\nExcluding components: {component_indices}")
    ica.exclude.extend(component_indices)
    print(f"Total excluded components: {ica.exclude}")
    
    return ica


def apply_ica_removal(raw, ica):
    """
    Apply ICA to remove artifact components from raw data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG data
    ica : mne.preprocessing.ICA
        ICA object with excluded components
    
    Returns
    -------
    raw : mne.io.Raw
        Cleaned raw data
    """
    print("\nApplying ICA to remove artifacts...")
    raw = ica.apply(raw)
    print("ICA applied successfully")
    
    return raw


# =============================================================================
# SAVE DATA
# =============================================================================

def save_cleaned_data(raw, events, subject, output_folder, stimulus_type):
    """Save ICA-cleaned data and events."""
    # Create output directory if needed
    output_dir = os.path.join(output_folder, subject, 'meg')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save events
    event_file = os.path.join(output_dir, f'{subject}-eve_{stimulus_type}.fif')
    write_events(event_file, events)
    
    # Save ICA-cleaned raw data
    raw_file = os.path.join(
        output_dir, 
        f'{subject}_resampled_300Hz-1-100Hz_filtered-ICA-eyeblink_{stimulus_type}.fif'
    )
    raw.save(raw_file, overwrite=True)
    
    print(f"\nSaved events to: {event_file}")
    print(f"Saved ICA-cleaned data to: {raw_file}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Run the complete preprocessing pipeline with ICA."""
    
    print("=" * 70)
    print("MEG PREPROCESSING PIPELINE WITH ICA")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1/7] Loading raw data...")
    raw, events, fs = load_raw_data(subject, MEG_folder, ch_types)
    
    # 2. Remove incomplete trials
    print("\n[2/7] Removing incomplete trials...")
    events_clean = remove_triggers(events)
    
    # 3. Annotate bad segments
    print("\n[3/7] Annotating bad segments...")
    raw = annotate_task_periods(raw, events_clean, fs)
    
    # 4. Resample and filter
    print("\n[4/7] Resampling and filtering...")
    raw, events_resampled, picks = resample_and_filter(
        raw, events_clean, fs, DESIRED_SFREQ, L_FREQ, H_FREQ, NOTCH_FREQS
    )
    
    # 5. Fit ICA
    print("\n[5/7] Fitting ICA...")
    ica = fit_ica(
        raw, 
        picks,
        method=ICA_METHOD,
        n_components=N_COMPONENTS,
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER
    )
    
    # 6. Visualize and select components
    print("\n[6/7] Visualizing ICA components...")
    visualize_ica_components(ica, raw)
    
    # Select artifact components
    print("\n" + "=" * 70)
    print("MANUAL STEP REQUIRED:")
    print("Inspect the ICA component plots and update 'artifact_components'")
    print("with the indices of components showing artifacts")
    print("=" * 70)
    
    artifact_components = [12, 13]  # UPDATE THESE based on visual inspection
    ica = select_artifact_components(ica, artifact_components)
    
    # Apply ICA
    raw = apply_ica_removal(raw, ica)
    
    # 7. Save cleaned data
    print("\n[7/7] Saving ICA-cleaned data...")
    save_cleaned_data(raw, events_resampled, subject, Output_MEG_folder, STIMULUS_TYPE)
    
    # Optional: Plot final data for inspection
    raw.plot(events_resampled)
    
    print("\n" + "=" * 70)
    print("PREPROCESSING PIPELINE COMPLETE!")
    print("=" * 70)


# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':
    main()