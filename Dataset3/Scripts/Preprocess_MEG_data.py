#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEG Complete Preprocessing Pipeline with Interactive ICA
Preprocesses raw MEG data and applies ICA with manual component selection
Author: filiztezcan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.io import read_raw_ctf
from mne import (find_events, Epochs, Annotations, write_events,
                 pick_types, read_epochs, read_events)
from mne.preprocessing import ICA

# ============================================================================
# CONFIGURATION
# ============================================================================

# Subject to process
SUBJECT = 'sub-006'  # <--- CHANGE THIS TO PROCESS A DIFFERENT SUBJECT

# Directory paths
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
MEG_FOLDER = os.path.join(parent_dir, 'raw')
OUTPUT_MEG_FOLDER = os.path.join(parent_dir, 'processed')

# Trigger definitions
TRIGGER_CODES = {
    'resting_state': 66,
    'controlsentence': 10,
    'semanticallyanomalous': 20,
    'syntacticallyanomalous': 30,
    'lexicosemanticgrouping': 40,
    'fixedwordorders': 50,
    'randomsyllables': 70
}

CONDITION_TRIGGERS = [10, 20, 30, 40, 50, 70]

# Channel type reassignments
CHANNEL_TYPES = {
    'EEG057-4302': 'eog',
    'EEG058-4302': 'eog',
    'EEG059-4302': 'ecg',
    'UPPT001': 'stim',
    'UPPT002': 'resp',
}

# Processing parameters
DESIRED_SFREQ = 300
NOTCH_FREQS = (50, 100)
HIGHPASS_FREQ = 0.3
LOWPASS_FREQ = 145

# Epoching parameters
EPOCH_TMIN = -2.0
EPOCH_TMAX = 28.8

# ICA parameters
ICA_METHOD = 'fastica'
ICA_N_COMPONENTS = 60
ICA_MAX_ITER = 10000
ICA_RANDOM_STATE = 42


# ============================================================================
# SUBJECT-SPECIFIC BAD SEGMENTS
# ============================================================================

def get_subject_bad_segments(subject):
    """Define subject-specific bad data segments."""
    bad_starts = []
    bad_ends = []
    event_deletions = []
    
    if subject == 'sub-006':
        bad_starts.extend([None])
        bad_ends.extend([None])
        event_deletions.append((229, 266))
    
    elif subject == 'sub-009':
        bad_starts.extend([None, 3303])
        bad_ends.extend([None, 3790])
        event_deletions.append((300, 306))
    
    elif subject == 'sub-017':
        bad_starts.extend([1776, 1796, 1784])
        bad_ends.extend([1778, 1840, 1790])
    
    elif subject == 'sub-018':
        bad_starts.extend([2078])
        bad_ends.extend([2079])
    
    elif subject == 'sub-021':
        bad_starts.extend([3512])
        bad_ends.extend([3520])
    
    elif subject == 'sub-024':
        bad_starts.extend([443])
        bad_ends.extend([446])
    
    elif subject == 'sub-031':
        bad_starts.extend([4066])
        bad_ends.extend([4068])
    
    elif subject == 'sub-036':
        bad_starts.extend([1002, 2426, 3037, 3177])
        bad_ends.extend([1004, 2428, 3041, 3179])
    
    elif subject == 'sub-037':
        bad_starts.extend([1896, 4274])
        bad_ends.extend([1898, 4276])
    
    return bad_starts, bad_ends, event_deletions


# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================

def load_raw_meg_data(subject):
    """Load raw MEG data for a subject."""
    meg_path = os.path.join(MEG_FOLDER, subject, 'ses-meg01/meg')
    ds_data = os.listdir(meg_path)[0]
    raw_file_name = os.path.join(meg_path, ds_data)
    
    print(f"\nLoading raw data: {raw_file_name}")
    
    raw = read_raw_ctf(raw_file_name, preload=True)
    raw = raw.set_channel_types(CHANNEL_TYPES)
    
    events = find_events(raw, shortest_event=1)
    fs = raw.info['sfreq']
    
    return raw, events, fs


# ============================================================================
# STEP 2: BAD SEGMENT ANNOTATION
# ============================================================================

def create_block_boundary_annotations(raw, events):
    """Create annotations for data segments between experimental blocks."""
    fs = raw.info['sfreq']
    block_empty_start = []
    block_empty_end = []
    
    # Before first and after last trigger
    block_empty_start.append(0)
    block_empty_end.append(events[0, 0] / fs - 2.1)
    
    block_empty_start.append(events[-1, 0] / fs + 3)
    block_empty_end.append(raw.n_times / fs)
    
    # Process events to find block boundaries
    for i in range(len(events) - 1):
        if events[i, 2] == 80:
            block_empty_start.append(events[i, 0] / fs - 2)
        
        if 179 < events[i, 2] < 210:
            j = i
            found = False
            while not found and j < len(events) - 1:
                j += 1
                if events[j, 2] in CONDITION_TRIGGERS:
                    found = True
                    block_empty_end.append(events[j, 0] / fs - 2.1)
                    break
        
        if i > 0 and events[i, 2] == 66:
            block_empty_end.append(events[i, 0] / fs - 1)
            block_empty_start.append(events[i - 1, 0] / fs + 3)
    
    return block_empty_start, block_empty_end


def annotate_bad_segments(raw, events, subject):
    """Annotate all bad data segments."""
    fs = raw.info['sfreq']
    
    block_starts, block_ends = create_block_boundary_annotations(raw, events)
    bad_starts, bad_ends, event_deletions = get_subject_bad_segments(subject)
    
    # Process subject-specific event-based bad segments
    for deletion_range in event_deletions:
        start_idx, end_idx = deletion_range
        block_starts.append(events[start_idx, 0] / fs - 1)
        block_ends.append(events[end_idx, 0] / fs - 2.1)
        events = np.delete(events, slice(start_idx, end_idx), 0)
    
    # Add subject-specific time-based bad segments
    for start, end in zip(bad_starts, bad_ends):
        if start is not None and end is not None:
            block_starts.append(start)
            block_ends.append(end)
    
    block_durations = [end - start for start, end in zip(block_starts, block_ends)]
    descriptions = ['bad'] * len(block_starts)
    annotations = Annotations(block_starts, block_durations, descriptions)
    
    raw = raw.set_annotations(annotations)
    
    return raw, events


# ============================================================================
# STEP 3: FILTERING AND RESAMPLING
# ============================================================================

def resample_and_adjust_events(raw, events, target_sfreq=300):
    """Resample raw data and adjust event timing."""
    original_sfreq = raw.info['sfreq']
    
    raw = raw.resample(target_sfreq, n_jobs=-1)
    
    events_new = events.copy()
    scaling_factor = target_sfreq / original_sfreq
    events_new[:, 0] = (events_new[:, 0] * scaling_factor).astype(int)
    
    return raw, events_new


def apply_notch_filter(raw, freqs=(50, 100)):
    """Apply notch filter to remove line noise."""
    return raw.notch_filter(freqs=freqs, n_jobs=-1)


# ============================================================================
# STEP 4: EPOCHING
# ============================================================================

def create_condition_epochs(raw, events):
    """Create epochs for experimental conditions."""
    event_dict = {
        'controlsentence': 10,
        'semanticallyanomalous': 20,
        'syntacticallyanomalous': 30,
        'lexicosemanticgrouping': 40,
        'fixedwordorders': 50,
        'randomsyllables': 70,
    }
    
    epochs = Epochs(
        raw, events,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=None,
        event_id=event_dict,
        preload=True,
        detrend=1
    )
    
    return epochs


def apply_bandpass_filter(epochs, l_freq=0.5, h_freq=145):
    """Apply bandpass filter to epochs."""
    return epochs.filter(
        l_freq, h_freq,
        method='fir',
        fir_window='hamming',
        fir_design='firwin',
        skip_by_annotation=('edge', 'bad_task', 'bad_between_bloks', 'bad'),
        n_jobs=-1
    )


# ============================================================================
# STEP 5: ICA WITH MANUAL COMPONENT SELECTION
# ============================================================================

def prepare_epochs_for_ica(epochs):
    """Pick MEG channels for ICA."""
    picks = pick_types(
        epochs.info,
        meg=True,
        ref_meg=False,
        eeg=False,
        eog=False,
        stim=False
    )
    epochs_ica = epochs.copy().pick(picks)
    return epochs_ica


def fit_ica(epochs):
    """Fit ICA to epochs data."""
    print("\nFitting ICA...")
    print(f"  Method: {ICA_METHOD}")
    print(f"  Components: {ICA_N_COMPONENTS}")
    print(f"  Max iterations: {ICA_MAX_ITER}")
    
    ica = ICA(
        method=ICA_METHOD,
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER,
        n_components=ICA_N_COMPONENTS
    )
    
    ica.fit(epochs, decim=None)
    print("  ICA fitting complete!")
    
    return ica


def interactive_component_selection(ica, epochs):
    """
    Display ICA components and get user input for exclusion.
    
    Returns list of component indices to exclude.
    """
    print("\n" + "="*80)
    print("INTERACTIVE ICA COMPONENT SELECTION")
    print("="*80)
    
    # Plot components
    print("\nDisplaying ICA components...")
    ica.plot_components()
    
    print("\nDisplaying component time courses...")
    ica.plot_sources(epochs, show_scrollbars=False)
    
    # Get user input
    print("\n" + "="*80)
    print("Please examine the ICA components in the plot windows.")
    print("Identify components that represent artifacts (eye blinks, muscle, etc.)")
    print("="*80)
    
    while True:
        user_input = input("\nEnter component indices to exclude (comma-separated, e.g., '0,1,22,23'): ")
        
        try:
            # Parse input
            if user_input.strip() == "":
                exclude_inds = []
            else:
                exclude_inds = [int(x.strip()) for x in user_input.split(',')]
            
            # Validate indices
            if all(0 <= idx < ICA_N_COMPONENTS for idx in exclude_inds):
                break
            else:
                print(f"ERROR: Component indices must be between 0 and {ICA_N_COMPONENTS-1}")
        except ValueError:
            print("ERROR: Invalid input. Please enter comma-separated integers (e.g., '0,1,22,23')")
    
    print(f"\nComponents to exclude: {exclude_inds}")
    return exclude_inds


def apply_ica(ica, epochs, exclude_inds):
    """Apply ICA to remove selected components."""
    ica.exclude.extend(exclude_inds)
    epochs_clean = ica.apply(epochs.copy())
    return epochs_clean


# ============================================================================
# STEP 6: SAVING
# ============================================================================

def save_final_data(epochs, events, subject, output_folder):
    """Save ICA-cleaned epochs with equalized event counts."""
    output_path = os.path.join(output_folder, subject, 'meg')
    os.makedirs(output_path, exist_ok=True)
    
    # Equalize event counts across conditions
    print("\nEqualizing event counts across conditions...")
    epochs_equalized = epochs.equalize_event_counts()[0]
    
    print(f"  Original epochs: {len(epochs)}")
    print(f"  Equalized epochs: {len(epochs_equalized)}")
    
    # Save epochs
    epochs_fname = os.path.join(
        output_path,
        f'{subject}_resampled_300Hz-05-145Hz_filtered-ICA-cleaned-epochs.fif'
    )
    epochs_equalized.save(epochs_fname, overwrite=True)
    print(f"\nSaved final epochs: {epochs_fname}")
    
    # Save events
    events_fname = os.path.join(output_path, f'{subject}-eve.fif')
    write_events(events_fname, events, overwrite=True)
    print(f"Saved events: {events_fname}")
    
    return epochs_equalized


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_complete_pipeline(subject):
    """
    Complete preprocessing pipeline from raw data to ICA-cleaned epochs.
    
    Pipeline steps:
    1. Load raw MEG data
    2. Annotate bad segments
    3. Resample to 300 Hz
    4. Apply notch filter
    5. Create epochs
    6. Apply bandpass filter
    7. Fit ICA
    8. Manual component selection (INTERACTIVE)
    9. Apply ICA to remove artifacts
    10. Save final cleaned data
    """
    print("\n" + "="*80)
    print("MEG PREPROCESSING PIPELINE WITH INTERACTIVE ICA")
    print("="*80)
    print(f"\nSubject: {subject}")
    print(f"Input folder: {MEG_FOLDER}")
    print(f"Output folder: {OUTPUT_MEG_FOLDER}")
    print("="*80)
    
    # STEP 1: Load data
    print("\n[STEP 1/10] Loading raw MEG data...")
    raw, events, original_sfreq = load_raw_meg_data(subject)
    print(f"  Original sampling rate: {original_sfreq} Hz")
    print(f"  Number of events: {len(events)}")
    
    # STEP 2: Annotate bad segments
    print("\n[STEP 2/10] Annotating bad segments...")
    raw, events = annotate_bad_segments(raw, events, subject)
    print(f"  Number of annotations: {len(raw.annotations)}")
    print(f"  Events after cleaning: {len(events)}")
    
    # STEP 3: Resample
    print(f"\n[STEP 3/10] Resampling to {DESIRED_SFREQ} Hz...")
    raw, events = resample_and_adjust_events(raw, events, DESIRED_SFREQ)
    
    # STEP 4: Notch filter
    print(f"\n[STEP 4/10] Applying notch filter at {NOTCH_FREQS} Hz...")
    raw = apply_notch_filter(raw, NOTCH_FREQS)
    
    # STEP 5: Create epochs
    print(f"\n[STEP 5/10] Creating epochs ({EPOCH_TMIN} to {EPOCH_TMAX} s)...")
    epochs = create_condition_epochs(raw, events)
    print(f"  Number of epochs: {len(epochs)}")
    
    # STEP 6: Bandpass filter
    print(f"\n[STEP 6/10] Applying bandpass filter ({HIGHPASS_FREQ}-{LOWPASS_FREQ} Hz)...")
    epochs = apply_bandpass_filter(epochs, HIGHPASS_FREQ, LOWPASS_FREQ)
    
    # Optional: Plot epochs for inspection
    print("\nDisplaying epochs for inspection...")
    epochs.plot(n_epochs=5, n_channels=30, scalings='auto')
    
    # STEP 7: Prepare for ICA
    print("\n[STEP 7/10] Preparing epochs for ICA (selecting MEG channels)...")
    epochs_ica = prepare_epochs_for_ica(epochs)
    print(f"  Number of MEG channels: {len(epochs_ica.ch_names)}")
    
    # STEP 8: Fit ICA
    print("\n[STEP 8/10] Fitting ICA...")
    ica = fit_ica(epochs_ica)
    
    # STEP 9: Interactive component selection
    print("\n[STEP 9/10] Interactive component selection...")
    exclude_inds = interactive_component_selection(ica, epochs_ica)
    
    # STEP 10: Apply ICA
    print(f"\n[STEP 10/10] Applying ICA to remove {len(exclude_inds)} components...")
    epochs_clean = apply_ica(ica, epochs, exclude_inds)
    
    # STEP 11: Save final data
    print("\n[STEP 11/10] Saving final ICA-cleaned data...")
    epochs_final = save_final_data(epochs_clean, events, subject, OUTPUT_MEG_FOLDER)
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nSubject: {subject}")
    print(f"Final epochs: {len(epochs_final)}")
    print(f"Excluded ICA components: {exclude_inds}")
    print(f"Output directory: {os.path.join(OUTPUT_MEG_FOLDER, subject, 'meg')}")
    print("="*80 + "\n")
    
    return epochs_final, ica, exclude_inds


# ============================================================================
# RUN PIPELINE
# ============================================================================

if __name__ == "__main__":
    try:
        epochs_final, ica, excluded_components = run_complete_pipeline(SUBJECT)
        print("\n✓ Pipeline executed successfully!")
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        raise