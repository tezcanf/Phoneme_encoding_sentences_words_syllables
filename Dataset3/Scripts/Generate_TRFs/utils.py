"""
Utility functions for MEG source localization and preprocessing.
"""
import os
from pathlib import Path
from typing import List, Tuple, Any
from functools import reduce

import numpy as np
import mne
from mne.minimum_norm import (
    read_inverse_operator,
    apply_inverse_epochs,
    make_inverse_operator,
)
import eelbrain

from config import SOURCE_PARAMS


def make_source_space(
    condition: str,
    subject: str,
    meg_dir: Path,
    subjects_dir: Path,
    stimuli_subject: List[str],
    gammatone: List[Any],
    tmax: float,
    nsamples: int
) -> Tuple[List[Any], List[Any]]:
    """
    Create source space representations for MEG data in the Superior Temporal Gyrus (STG).
    
    This function:
    1. Loads or creates forward and inverse operators
    2. Applies inverse solution to epochs
    3. Morphs to fsaverage space
    4. Extracts activity from STG
    5. Converts to eelbrain NDVar format
    
    Args:
        condition: Experimental condition name
        subject: Subject ID
        meg_dir: Directory containing MEG data
        subjects_dir: FreeSurfer subjects directory
        stimuli_subject: List of stimulus names
        gammatone: Gammatone spectrograms (for time reference)
        tmax: Maximum time for cropping
        nsamples: Expected number of time samples
        
    Returns:
        Tuple of (left hemisphere source data, right hemisphere source data)
    """
    # Fixed to STG region
    region = 'STG'
    # File paths
    fname_fsaverage_src = os.path.join(
        subjects_dir, 'fsaverage', 'bem', 'fsaverage-ico-4-src.fif'
    )
    forward_fname = os.path.join(meg_dir, f'{subject}_forward_ICA_all.fif')
    inverse_fname = os.path.join(meg_dir, f'{subject}_inverse_ICA_all.fif')
    epochs_fname = os.path.join(
        meg_dir, 
        f'{subject}_resampled_300Hz-05-145Hz_filtered-ICA-eyeblink-epochs.fif'
    )
    
    # Load epochs
    epochs = mne.read_epochs(epochs_fname, preload=True)
    info = epochs.info
    
    # Create or load forward solution
    if not os.path.isfile(forward_fname):
        print(f"Creating forward solution for {subject}...")
        fwd = _create_forward_solution(subject, info, subjects_dir, forward_fname)
    else:
        print(f"Loading forward solution for {subject}...")
        fwd = mne.read_forward_solution(forward_fname)
    
    # Create or load inverse operator
    if not os.path.isfile(inverse_fname):
        print(f"Creating inverse operator for {subject}...")
        inverse_operator = _create_inverse_operator(
            epochs, info, fwd, inverse_fname
        )
    else:
        print(f"Loading inverse operator for {subject}...")
        inverse_operator = read_inverse_operator(inverse_fname)
    
    print('Inverse operator ready')
    
    # Get STG labels for both hemispheres
    stc_lh_merged_label, stc_rh_merged_label = _get_region_labels(subjects_dir)
    
    # Load source space and add distances
    src_to = mne.read_source_spaces(fname_fsaverage_src)
    src_to = mne.add_source_space_distances(src_to)
    
    # Process each stimulus
    stc_lh_all = []
    stc_rh_all = []
    
    # Get epoch name
    from config import CONDITION_TO_EPOCH_NAME
    epoch_name = CONDITION_TO_EPOCH_NAME.get(condition)
    if epoch_name is None:
        epoch_name = ''.join(condition.split('_'))
    
    for i in range(len(stimuli_subject)):
        print(f"Processing stimulus {i+1}/{len(stimuli_subject)}...")
        
        # Crop and filter epoch
        raw_cropped = epochs[epoch_name][i].crop(
            SOURCE_PARAMS['crop_tmin'],
            SOURCE_PARAMS['crop_tmax'] + tmax
        )
        raw_cropped = raw_cropped.filter(
            l_freq=None,
            h_freq=SOURCE_PARAMS['filter_highpass']
        )
        
        # Apply inverse solution
        stc = apply_inverse_epochs(
            raw_cropped,
            inverse_operator,
            lambda2=1.0 / (3.0 ** 2),
            method="dSPM"
        )
        stc = stc[0]
        
        # Morph to fsaverage
        morph = mne.compute_source_morph(
            stc,
            subject_from=subject,
            subject_to='fsaverage',
            src_to=src_to,
            subjects_dir=subjects_dir
        )
        stc = morph.apply(stc)
        
        # Resample
        stc = stc.resample(SOURCE_PARAMS['resample_freq'])
        
        # Extract region of interest
        stc_lh = stc.in_label(stc_lh_merged_label)
        stc_rh = stc.in_label(stc_rh_merged_label)
        
        # Ensure correct number of samples
        stc_lh, stc_rh = _adjust_samples(stc_lh, stc_rh, nsamples)
        
        # Convert to eelbrain NDVar
        stc_lh_ndvar = eelbrain.load.fiff.stc_ndvar(
            stc=stc_lh,
            subject='fsaverage',
            src='ico-4',
            subjects_dir=subjects_dir,
            parc='aparc',
            check=True
        )
        stc_rh_ndvar = eelbrain.load.fiff.stc_ndvar(
            stc=stc_rh,
            subject='fsaverage',
            src='ico-4',
            subjects_dir=subjects_dir,
            parc='aparc',
            check=True
        )
        
        stc_lh_all.append(stc_lh_ndvar)
        stc_rh_all.append(stc_rh_ndvar)
        
        del raw_cropped, stc
    
    return stc_lh_all, stc_rh_all


def _create_forward_solution(
    subject: str,
    info: mne.Info,
    subjects_dir: Path,
    output_fname: str
) -> mne.Forward:
    """Create and save forward solution."""
    # Setup source space
    src = mne.setup_source_space(
        subject,
        spacing='ico4',
        subjects_dir=subjects_dir
    )
    
    # Create BEM model
    conductivity = (0.3,)  # Single layer
    model = mne.make_bem_model(
        subject=subject,
        conductivity=conductivity,
        subjects_dir=subjects_dir
    )
    bem = mne.make_bem_solution(model)
    
    # Get transformation
    trans = os.path.join(subjects_dir, subject, 'meg', f'{subject}-trans.fif')
    
    # Create forward solution
    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=-1,
        verbose=True
    )
    
    # Convert to surface orientation
    fwd = mne.convert_forward_solution(fwd, surf_ori=True)
    
    # Save
    mne.write_forward_solution(output_fname, fwd, overwrite=True)
    
    return fwd


def _create_inverse_operator(
    epochs: mne.Epochs,
    info: mne.Info,
    fwd: mne.Forward,
    output_fname: str
) -> mne.minimum_norm.InverseOperator:
    """Create and save inverse operator."""
    # Compute noise covariance
    noise_cov = mne.compute_covariance(
        epochs,
        tmin=4.9,
        tmax=5.9,
        method='empirical',
        rank='info',
        verbose=True
    )
    
    # Create inverse operator
    inverse_operator = make_inverse_operator(
        info,
        fwd,
        noise_cov,
        fixed=True
    )
    
    # Save
    mne.minimum_norm.write_inverse_operator(output_fname, inverse_operator)
    
    return inverse_operator


def _get_region_labels(
    subjects_dir: Path
) -> Tuple[mne.Label, mne.Label]:
    """
    Get left and right hemisphere labels for Superior Temporal Gyrus (STG).
    
    Args:
        subjects_dir: FreeSurfer subjects directory
        
    Returns:
        Tuple of (left hemisphere STG label, right hemisphere STG label)
    """
    labels = mne.read_labels_from_annot(
        subject='fsaverage',
        parc='aparc.a2009s',
        subjects_dir=subjects_dir
    )
    labels_name = [l.name for l in labels]
    
    # STG labels
    search_lh = [
        'G_temp_sup-G_T_transv-lh',
        'G_temp_sup-Lateral-lh',
        'G_temp_sup-Plan_polar-lh',
        'G_temp_sup-Plan_tempo-lh'
    ]
    search_rh = [
        'G_temp_sup-G_T_transv-rh',
        'G_temp_sup-Lateral-rh',
        'G_temp_sup-Plan_polar-rh',
        'G_temp_sup-Plan_tempo-rh'
    ]
    
    # Get label indices
    label_index_lh = [labels_name.index(l) for l in search_lh]
    label_index_rh = [labels_name.index(l) for l in search_rh]
    
    # Extract and merge labels
    lh_label_list = [labels[i] for i in label_index_lh]
    rh_label_list = [labels[i] for i in label_index_rh]
    
    stc_lh_merged_label = reduce(lambda x, y: x + y, lh_label_list)
    stc_rh_merged_label = reduce(lambda x, y: x + y, rh_label_list)
    
    return stc_lh_merged_label, stc_rh_merged_label


def _adjust_samples(
    stc_lh: mne.SourceEstimate,
    stc_rh: mne.SourceEstimate,
    target_nsamples: int
) -> Tuple[mne.SourceEstimate, mne.SourceEstimate]:
    """
    Adjust number of samples in source estimates to match target.
    
    Args:
        stc_lh: Left hemisphere source estimate
        stc_rh: Right hemisphere source estimate
        target_nsamples: Target number of samples
        
    Returns:
        Tuple of adjusted source estimates
    """
    current_samples = len(stc_lh._data.T)
    crop = current_samples - target_nsamples
    
    if crop != 0:
        print(f'Adjusting samples: removing {crop} time points')
        
    for _ in range(crop):
        stc_lh._data = np.delete(stc_lh._data, 0, -1)
        stc_lh._times = np.delete(stc_lh.times, -1)
        stc_rh._data = np.delete(stc_rh._data, 0, -1)
        stc_rh._times = np.delete(stc_rh.times, -1)
    
    return stc_lh, stc_rh
