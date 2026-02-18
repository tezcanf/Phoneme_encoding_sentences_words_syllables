#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Utility Functions for TRF Estimation

This module contains functions for source space reconstruction that work
with both Dutch and Chinese stimuli by accepting a stimulus_type parameter.

@author: filiztezcan
"""

import os
from functools import reduce
import numpy as np
import mne
from mne.minimum_norm import (read_inverse_operator, apply_inverse_raw, 
                               make_inverse_operator, write_inverse_operator)
from mne import pick_types
import eelbrain


def make_source_space(condition, subject, processed_MEG_folder, subjects_dir, 
                     STIMULI, stimulus_type='Dutch'):
    """
    Create source space data for TRF estimation
    
    This function processes MEG data to create source-localized activity
    in bilateral superior temporal gyrus regions for TRF estimation.
    
    Parameters
    ----------
    condition : str
        Condition name ('Words' or 'Syllables')
    subject : str
        Subject ID (e.g., 'sub-003')
    processed_MEG_folder : Path
        Path to processed MEG data directory
    subjects_dir : str
        Path to FreeSurfer subjects directory
    STIMULI : list
        List of stimulus identifiers (kept for compatibility, not used)
    stimulus_type : str, optional
        Type of stimuli ('Dutch' or 'Chinese'), default is 'Dutch'
        This determines which MEG and events files to load
    
    Returns
    -------
    stc_lh_all : list of NDVar
        List of source space data for left hemisphere (one per trial)
    stc_rh_all : list of NDVar
        List of source space data for right hemisphere (one per trial)
    """
    
    # Parameters
    tmax = 0.7  # Maximum time for cropping
    fname_fsaverage_src = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                       'fsaverage-ico-4-src.fif')
    
    # File paths - dependent on stimulus type
    raw_file_name = os.path.join(processed_MEG_folder, 
                                 f'{subject}_resampled_300Hz-1-100Hz_filtered-ICA-eyeblink_{stimulus_type}_stimuli.fif')
    frwfname = os.path.join(processed_MEG_folder, f'{subject}_forward.fif')
    invfname = os.path.join(processed_MEG_folder, f'{subject}_inverse.fif')
    events_file = os.path.join(subjects_dir, subject, 'meg', 
                               f'{subject}-eve_{stimulus_type}_stimuli.fif')
    
    # Verify files exist
    if not os.path.exists(raw_file_name):
        raise FileNotFoundError(f"MEG raw file not found: {raw_file_name}")
    if not os.path.exists(events_file):
        raise FileNotFoundError(f"Events file not found: {events_file}")
    
    # Load data
    print(f"    Loading MEG data: {os.path.basename(raw_file_name)}")
    raw = mne.io.read_raw_fif(raw_file_name, preload=True, verbose=False)
    info = raw.info 
    
    print(f"    Loading events: {os.path.basename(events_file)}")
    events = mne.read_events(events_file, verbose=False)
    
    # Create or load forward solution
    if not os.path.isfile(frwfname):
        print("    Creating forward solution...")
        src = mne.setup_source_space(subject, spacing='ico4',
                                      subjects_dir=subjects_dir, verbose=False)
        
        # Single layer BEM
        conductivity = (0.3,)
        model = mne.make_bem_model(subject=subject, 
                                    conductivity=conductivity,
                                    subjects_dir=subjects_dir,
                                    verbose=False)
        bem = mne.make_bem_solution(model, verbose=False)
        
        trans = os.path.join(subjects_dir, subject, 'meg', f'{subject}-trans.fif')
        
        if not os.path.exists(trans):
            raise FileNotFoundError(f"Transformation file not found: {trans}")
        
        fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem,
                                        meg=True, eeg=False, mindist=5.0, 
                                        n_jobs=-1, verbose=False)
        
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, verbose=False)
        mne.write_forward_solution(frwfname, fwd, overwrite=True, verbose=False)
        print(f"    ✓ Forward solution saved: {os.path.basename(frwfname)}")
    else:
        fwd = mne.read_forward_solution(frwfname, verbose=False)
        print(f"    ✓ Forward solution loaded: {os.path.basename(frwfname)}")
    
    # Create or load inverse operator
    if not os.path.isfile(invfname):
        print("    Creating inverse operator...")
        event_id = [10, 20, 30]
        baseline = (None, None)
        picks = pick_types(raw.info, meg=True, ref_meg=False, eeg=False, 
                          eog=False, stim=False)
        
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=5.95, 
                           preload=True, baseline=baseline, picks=picks,
                           verbose=False)
        
        # Compute noise covariance from silent period (4.9-5.9s)
        noise_cov = mne.compute_covariance(epochs, tmin=4.9, tmax=5.9, 
                                          method='empirical', rank='info', 
                                          verbose=False)
        
        inverse_operator = make_inverse_operator(info, fwd, noise_cov, 
                                                 fixed=True, verbose=False)
        write_inverse_operator(invfname, inverse_operator, verbose=False)
        print(f"    ✓ Inverse operator saved: {os.path.basename(invfname)}")
    else: 
        inverse_operator = read_inverse_operator(invfname, verbose=False)
        print(f"    ✓ Inverse operator loaded: {os.path.basename(invfname)}")
    
    # Inverse parameters
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    method = "dSPM"
    
    raw = raw.pick_types(meg=True, ref_meg=False)
    
    # Define ROI labels (bilateral superior temporal gyrus)
    print("    Loading anatomical labels...")
    labels = mne.read_labels_from_annot(subject='fsaverage', parc='aparc.a2009s', 
                                        subjects_dir=subjects_dir, verbose=False)
    labels_name = [l.name for l in labels]
    
    # Superior temporal gyrus regions
    search_lh = np.array(['G_temp_sup-G_T_transv-lh', 'G_temp_sup-Lateral-lh', 
                         'G_temp_sup-Plan_polar-lh', 'G_temp_sup-Plan_tempo-lh'])
    search_rh = np.array(['G_temp_sup-G_T_transv-rh', 'G_temp_sup-Lateral-rh', 
                         'G_temp_sup-Plan_polar-rh', 'G_temp_sup-Plan_tempo-rh'])

    # Merge labels for each hemisphere
    label_index_lh = [labels_name.index(l) for l in search_lh]
    label_index_rh = [labels_name.index(l) for l in search_rh]
    
    lh_label_list = [labels[i] for i in label_index_lh]
    rh_label_list = [labels[i] for i in label_index_rh]
    
    stc_lh_merged_label = reduce(lambda x, y: x + y, lh_label_list)
    stc_rh_merged_label = reduce(lambda x, y: x + y, rh_label_list)
    
    # Load fsaverage source space
    print('    Reading fsaverage source space...')
    src_to = mne.read_source_spaces(fname_fsaverage_src, verbose=False)
    
    # Select trials based on condition
    if condition == 'Words':
        onsets_all = np.where(events[:, 2] == 10)[0]
        onsets = onsets_all[:40].copy()
    elif condition == 'Syllables':
        onsets_all = np.where(events[:, 2] == 20)[0]
        onsets = onsets_all[:40].copy()
    else:
        raise ValueError(f"Unknown condition: {condition}. Must be 'Words' or 'Syllables'")
    
    print(f"    Processing {len(onsets)} trials for {condition} condition...")
    
    # Process each trial
    stc_lh_all = []
    stc_rh_all = []
    
    for t, index in enumerate(onsets):  
        start = events[index, 0]
        tstart = start / raw.info['sfreq']
        tend = tstart + 4
        
        if (t + 1) % 10 == 0:
            print(f'      Trial {t+1}/{len(onsets)}')
        
        # Crop raw data around trial
        raw_cropped = raw.copy().crop(tmin=tstart - 0.05, tmax=tend + tmax)
        raw_cropped.load_data()
        
        # Apply inverse solution
        stc = apply_inverse_raw(raw_cropped, inverse_operator, lambda2, method,
                               verbose=False)
        
        # Low-pass filter at 8.5 Hz
        stc._data = mne.filter.filter_data(stc.data, 300, l_freq=None, h_freq=8.5,
                                          verbose=False)
        
        # Morph to fsaverage
        morph = mne.compute_source_morph(stc, subject_from=subject,
                                         subject_to='fsaverage', src_to=src_to,
                                         subjects_dir=subjects_dir, verbose=False)
        stc = morph.apply(stc)
        
        # Resample to 100 Hz
        stc = stc.resample(100, verbose=False)
        
        # Extract ROI activity
        stc_lh = stc.in_label(stc_lh_merged_label) 
        stc_rh = stc.in_label(stc_rh_merged_label)
        
        del raw_cropped, stc
        
        # Crop to expected length (474 samples = 4.74s at 100 Hz)
        expected_samples = 474
        actual_samples = len(stc_lh._data.T)
        crop = actual_samples - expected_samples
        
        if crop != 0:
            if (t + 1) % 10 == 0:
                print(f'        Trimming {crop} samples (expected {expected_samples}, got {actual_samples})')
            stc_lh._data = stc_lh._data[:, :-crop]
            stc_lh._times = stc_lh.times[:-crop]
            stc_rh._data = stc_rh._data[:, :-crop]
            stc_rh._times = stc_rh.times[:-crop]
        
        # Convert to eelbrain NDVar
        stc_lh_ndvar = eelbrain.load.fiff.stc_ndvar(stc=stc_lh, subject='fsaverage',  
                                                     src='ico-4', subjects_dir=subjects_dir, 
                                                     parc='aparc', check=True)
        stc_rh_ndvar = eelbrain.load.fiff.stc_ndvar(stc=stc_rh, subject='fsaverage',  
                                                     src='ico-4', subjects_dir=subjects_dir, 
                                                     parc='aparc', check=True)
        
        stc_lh_all.append(stc_lh_ndvar)
        stc_rh_all.append(stc_rh_ndvar)
    
    print(f"    ✓ Processed all {len(onsets)} trials")
    
    return stc_lh_all, stc_rh_all
