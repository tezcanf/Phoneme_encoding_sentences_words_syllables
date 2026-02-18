"""
TRF Estimation Pipeline for MEG Data
This script estimates Temporal Response Functions (TRFs) for multiple models
across different experimental conditions (Sentences and Word_list).
"""

from pathlib import Path
import os
import copy
import numpy as np
import mne
import eelbrain
import trftools
from functools import reduce
from mne.io import read_raw_ctf
from mne.minimum_norm import (read_inverse_operator, apply_inverse_raw, 
                               make_inverse_operator)
from mne import pick_types

# ============================================================================
# Configuration
# ============================================================================

# Set number of workers for parallel processing
eelbrain.configure(n_workers=8)

# Paths
DATA_ROOT = Path.cwd().parents[1]
SUBJECTS_DIR = DATA_ROOT / 'processed'

# Analysis parameters
SUBJECT = 'sub-002'
CONDITIONS = ['Sentences', 'Word_list']
REGION = 'STG'  # Options: 'AC', 'STG', 'IFG', 'Whole_brain'
SNR = 3.0
METHOD = "dSPM"
SOURCE_SAVE = True  # Whether to save/load reconstructed source spaces

# TRF parameters
TMIN = -0.05
TMAX = 0.7
BASIS = 0.050
PARTITIONS = 5
TEST = 1


# ============================================================================
# Utility Functions
# ============================================================================

def get_stimulus_files(condition):
    """Get sorted list of stimulus files for a given condition."""
    stimulus_dir = DATA_ROOT / 'Materials' / 'Stimuli' / condition
    stimuli = [f.split('.')[0] for f in os.listdir(stimulus_dir) if f.endswith('wav')]
    stimuli.sort()
    
    # Remove missing triggers for specific subjects
    if SUBJECT == 'sub-009':
        trigger_map = {'Sentences': '101_42', 'Word_list': '105_42'}
        if trigger_map[condition] in stimuli:
            stimuli.remove(trigger_map[condition])
    
    if SUBJECT == 'sub-012':
        trigger_map = {'Sentences': '101_44', 'Word_list': '105_44'}
        if trigger_map[condition] in stimuli:
            stimuli.remove(trigger_map[condition])
    
    return stimuli


def load_data_per_story(raw, events, wav, condition, tmax=0.7, new_fs=100):
    """Extract MEG data for a specific story/trial."""
    trigger_map = {'Sentences': (113, 200), 'Word_list': (133, 200)}
    start_trigger, offset = trigger_map[condition]
    end_trigger = int(wav.split('_')[1]) + offset
    
    for i in range(len(events)):
        if events[i, 2] == start_trigger and events[i+1, 2] == end_trigger:
            tstart = events[i, 0] / raw.info['sfreq']
            tend = events[i+1, 0] / raw.info['sfreq']
            break
    
    raw_cropped = raw.copy().crop(tmin=tstart - 0.05, tmax=tend + tmax)
    raw_cropped.load_data()
    
    return raw_cropped


def get_brain_labels(region, labels, labels_name):
    """Get brain labels for the specified region."""
    region_map = {
        'AC': {
            'lh': ['G_temp_sup-G_T_transv-lh'],
            'rh': ['G_temp_sup-G_T_transv-rh']
        },
        'STG': {
            'lh': ['G_temp_sup-G_T_transv-lh', 'G_temp_sup-Lateral-lh', 
                   'G_temp_sup-Plan_polar-lh', 'G_temp_sup-Plan_tempo-lh'],
            'rh': ['G_temp_sup-G_T_transv-rh', 'G_temp_sup-Lateral-rh', 
                   'G_temp_sup-Plan_polar-rh', 'G_temp_sup-Plan_tempo-rh']
        },
        'IFG': {
            'lh': ['G_front_inf-Opercular-lh', 'G_front_inf-Orbital-lh', 
                   'G_front_inf-Triangul-lh'],
            'rh': ['G_front_inf-Opercular-rh', 'G_front_inf-Orbital-rh', 
                   'G_front_inf-Triangul-rh']
        },
        'Whole_brain': {
            'lh': [f for f in labels_name if f.endswith('-lh')],
            'rh': [f for f in labels_name if f.endswith('-rh')]
        }
    }
    
    search_lh = region_map[region]['lh']
    search_rh = region_map[region]['rh']
    
    label_index_lh = [labels_name.index(l) for l in search_lh]
    label_index_rh = [labels_name.index(l) for l in search_rh]
    
    lh_label_list = [labels[i] for i in label_index_lh]
    rh_label_list = [labels[i] for i in label_index_rh]
    
    stc_lh_merged_label = reduce(lambda x, y: x + y, lh_label_list)
    stc_rh_merged_label = reduce(lambda x, y: x + y, rh_label_list)
    
    return stc_lh_merged_label, stc_rh_merged_label


def get_source_dir(meg_dir, condition):
    """
    Get the directory for saving/loading source spaces.
    Creates the directory if it doesn't exist.
    
    Parameters:
    -----------
    meg_dir : Path
        Base MEG directory for the subject
    condition : str
        Experimental condition name
        
    Returns:
    --------
    Path : Directory for source space files
    """
    source_dir = meg_dir / 'source' / condition
    source_dir.mkdir(parents=True, exist_ok=True)
    return source_dir


def save_source_spaces(stc_lh_all, stc_rh_all, source_dir, stimuli, region):
    """
    Save reconstructed source spaces to disk.
    
    Parameters:
    -----------
    stc_lh_all : list
        List of left hemisphere source time courses
    stc_rh_all : list
        List of right hemisphere source time courses
    source_dir : Path
        Directory to save files
    stimuli : list
        List of stimulus names
    region : str
        Brain region name (for filename)
    """
    for i, stimulus in enumerate(stimuli):
        lh_path = source_dir / f'{stimulus}_{region}_lh.pickle'
        rh_path = source_dir / f'{stimulus}_{region}_rh.pickle'
        
        eelbrain.save.pickle(stc_lh_all[i], lh_path)
        eelbrain.save.pickle(stc_rh_all[i], rh_path)
    
    print(f"Saved {len(stimuli)} source spaces to {source_dir}")


def load_source_spaces(source_dir, stimuli, region):
    """
    Load previously saved source spaces from disk.
    
    Parameters:
    -----------
    source_dir : Path
        Directory containing saved files
    stimuli : list
        List of stimulus names
    region : str
        Brain region name (for filename)
        
    Returns:
    --------
    tuple : (stc_lh_all, stc_rh_all) if all files exist, otherwise (None, None)
    """
    stc_lh_all = []
    stc_rh_all = []
    
    # Check if all required files exist
    for stimulus in stimuli:
        lh_path = source_dir / f'{stimulus}_{region}_lh.pickle'
        rh_path = source_dir / f'{stimulus}_{region}_rh.pickle'
        
        if not lh_path.exists() or not rh_path.exists():
            return None, None
    
    # Load all files
    for stimulus in stimuli:
        lh_path = source_dir / f'{stimulus}_{region}_lh.pickle'
        rh_path = source_dir / f'{stimulus}_{region}_rh.pickle'
        
        stc_lh_all.append(eelbrain.load.unpickle(lh_path))
        stc_rh_all.append(eelbrain.load.unpickle(rh_path))
    
    print(f"Loaded {len(stimuli)} source spaces from {source_dir}")
    return stc_lh_all, stc_rh_all


def make_source_space(condition, subject, meg_dir, subjects_dir, stimuli, gammatone, save_sources=SOURCE_SAVE):
    """
    Reconstruct source space activity for all stimuli in a condition.
    
    Parameters:
    -----------
    condition : str
        Experimental condition name
    subject : str
        Subject identifier
    meg_dir : Path
        MEG data directory for the subject
    subjects_dir : str
        FreeSurfer subjects directory
    stimuli : list
        List of stimulus names
    gammatone : list
        List of gammatone predictors (for length matching)
    save_sources : bool
        Whether to save/load source spaces from disk
    
    Returns:
    --------
    stc_lh_all : list
        List of left hemisphere source time courses
    stc_rh_all : list
        List of right hemisphere source time courses
    """
    # Check if we should try to load saved source spaces
    if save_sources:
        source_dir = get_source_dir(meg_dir, condition)
        stc_lh_all, stc_rh_all = load_source_spaces(source_dir, stimuli, REGION)
        
        if stc_lh_all is not None and stc_rh_all is not None:
            print("Using saved source spaces")
            return stc_lh_all, stc_rh_all
        else:
            print("Saved source spaces not found, will reconstruct and save")
    
    fname_fsaverage_src = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                       'fsaverage-ico-4-src.fif')
    
    # Load raw MEG data
    raw_file_name = os.path.join(meg_dir, f'{subject}_resampled_300Hz-ICA-raw.fif')
    raw = mne.io.read_raw_fif(raw_file_name, preload=True)
    info = raw.info
    
    # Setup source space
    src = mne.setup_source_space(subject, spacing='ico4', subjects_dir=subjects_dir)
    
    # Create BEM model
    conductivity = (0.3,)  # Single layer
    model = mne.make_bem_model(subject=subject, conductivity=conductivity,
                                subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    
    # Create forward solution
    trans = os.path.join(subjects_dir, subject, 'meg', f'{subject}_trans.fif')
    fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem,
                                    meg=True, eeg=False, mindist=5.0, n_jobs=-1)
    
    # Load events
    events_file = os.path.join(subjects_dir, subject, 'meg', f'{subject}-eve.fif')
    events = mne.read_events(events_file)
    
    # Compute noise covariance
    event_id = [111, 112, 113, 114, 131, 132, 133, 134]
    baseline = (None, 0)
    picks = pick_types(raw.info, meg=True, ref_meg=False, eeg=False, 
                       eog=False, stim=False)
    epochs_noise = mne.Epochs(raw, events, event_id=event_id, tmin=-1.5, 
                              tmax=0, preload=True, baseline=baseline, picks=picks)
    noise_cov = mne.compute_covariance(epochs_noise, tmax=0., method='empirical',
                                       rank='info', verbose=True)
    
    # Create inverse operator
    fwd = mne.convert_forward_solution(fwd, surf_ori=True)
    inverse_operator = make_inverse_operator(info, fwd, noise_cov, fixed=True)
    
    lambda2 = 1.0 / SNR ** 2
    raw = raw.pick_types(meg=True, ref_meg=False)
    
    # Load brain labels
    labels = mne.read_labels_from_annot(subject='fsaverage', parc='aparc.a2009s',
                                        subjects_dir=subjects_dir)
    labels_name = [l.name for l in labels]
    stc_lh_merged_label, stc_rh_merged_label = get_brain_labels(REGION, labels, labels_name)
    
    # Setup morphing
    src_to = mne.read_source_spaces(fname_fsaverage_src)
    src_to = mne.add_source_space_distances(src_to)
    
    # Process each stimulus
    stc_lh_all = []
    stc_rh_all = []
    
    for i, wav in enumerate(stimuli):
        # Source reconstruction
        raw_cropped = load_data_per_story(raw, events, wav, condition)
        raw_cropped = raw_cropped.filter(l_freq=None, h_freq=50)
        
        stc = apply_inverse_raw(raw_cropped, inverse_operator, lambda2, METHOD)
        del raw_cropped
        
        # Morph to fsaverage
        morph = mne.compute_source_morph(stc, subject_from=subject,
                                         subject_to='fsaverage', src_to=src_to,
                                         subjects_dir=subjects_dir)
        stc = morph.apply(stc)
        stc = stc.resample(100)
        
        # Extract labels
        stc_lh = stc.in_label(stc_lh_merged_label)
        stc_rh = stc.in_label(stc_rh_merged_label)
        del stc
        
        # Adjust length to match predictors
        crop = len(stc_lh._data.T) - np.shape(gammatone[i])[1]
        if crop != 0:
            print(f'Adjusting length by {crop} samples')
            for _ in range(crop):
                stc_lh._data = np.delete(stc_lh._data, 0, -1)
                stc_lh._times = np.delete(stc_lh.times, -1)
                stc_rh._data = np.delete(stc_rh._data, 0, -1)
                stc_rh._times = np.delete(stc_rh.times, -1)
        
        # Convert to eelbrain NDVar
        stc_lh_ndvar = eelbrain.load.fiff.stc_ndvar(stc=stc_lh, subject='fsaverage',
                                                     src='ico-4', subjects_dir=subjects_dir,
                                                     parc='aparc', check=True)
        stc_rh_ndvar = eelbrain.load.fiff.stc_ndvar(stc=stc_rh, subject='fsaverage',
                                                     src='ico-4', subjects_dir=subjects_dir,
                                                     parc='aparc', check=True)
        
        stc_lh_all.append(stc_lh_ndvar)
        stc_rh_all.append(stc_rh_ndvar)
    
    # Save source spaces if requested
    if save_sources:
        source_dir = get_source_dir(meg_dir, condition)
        save_source_spaces(stc_lh_all, stc_rh_all, source_dir, stimuli, REGION)
    
    return stc_lh_all, stc_rh_all


# ============================================================================
# Predictor Loading and Processing
# ============================================================================

def load_predictors(condition, stimuli):
    """Load and process all predictors for a given condition."""
    predictor_dir = DATA_ROOT / 'Materials' / 'Predictors' / condition
    
    # Load gammatone spectrograms (reference time axis)
    gammatone = [eelbrain.load.unpickle(predictor_dir / f'{stimulus}~gammatone-8.pickle') 
                 for stimulus in stimuli]
    gammatone = [x.bin(0.01, dim='time', label='start') for x in gammatone]
    gammatone = [trftools.pad(x, tstart=-0.05, tstop=x.time.tstop + 0.7, name='gammatone') 
                 for x in gammatone]
    
    # Load onset spectrograms
    gammatone_onsets = [eelbrain.load.unpickle(predictor_dir / f'{stimulus}~gammatone-on-8.pickle') 
                        for stimulus in stimuli]
    gammatone_onsets = [x.bin(0.01, dim='time', label='start') for x in gammatone_onsets]
    gammatone_onsets = [eelbrain.set_time(x, gt.time, name='gammatone_on') 
                        for x, gt in zip(gammatone_onsets, gammatone)]
    
    # Load phoneme tables
    phoneme_tables = [eelbrain.load.unpickle(predictor_dir / f'{stimulus}~phoneme_cohort_model.pickle') 
                      for stimulus in stimuli]
    phoneme_onsets = [eelbrain.event_impulse_predictor(gt.time, ds=ds, name='phonemes') 
                      for gt, ds in zip(gammatone, phoneme_tables)]
    phoneme_surprisal = [eelbrain.event_impulse_predictor(gt.time, value='cohort_surprisal', 
                                                          ds=ds, name='cohort_surprisal') 
                         for gt, ds in zip(gammatone, phoneme_tables)]
    phoneme_entropy = [eelbrain.event_impulse_predictor(gt.time, value='cohort_entropy', 
                                                        ds=ds, name='cohort_entropy') 
                       for gt, ds in zip(gammatone, phoneme_tables)]
    
    # Load word tables (GPT-2 based)
    phoneme_tables_gpt = [eelbrain.load.unpickle(predictor_dir / f'{stimulus}~phoneme_cohort_model_GPT2_new_large.pickle') 
                          for stimulus in stimuli]
    word_surprisal = [eelbrain.event_impulse_predictor(gt.time, value='word_surprisal_GPT', 
                                                       ds=ds, name='word_surprisal') 
                      for gt, ds in zip(gammatone, phoneme_tables_gpt)]
    word_entropy = [eelbrain.event_impulse_predictor(gt.time, value='word_entropy_GPT', 
                                                     ds=ds, name='word_entropy') 
                    for gt, ds in zip(gammatone, phoneme_tables_gpt)]
    word_number_name = 'word_number' if condition == 'Word_list' else 'cohort_entropy'
    word_onset = [eelbrain.event_impulse_predictor(gt.time, value='word_number', 
                                                   ds=ds, name=word_number_name) 
                  for gt, ds in zip(gammatone, phoneme_tables_gpt)]
    
    # Create shuffled predictors for control analysis
    phoneme_surprisal_shuffled = shuffle_predictor(copy.deepcopy(phoneme_surprisal))
    phoneme_entropy_shuffled = shuffle_predictor(copy.deepcopy(phoneme_entropy))
    
    return {
        'gammatone': gammatone,
        'gammatone_onsets': gammatone_onsets,
        'phoneme_onsets': phoneme_onsets,
        'phoneme_surprisal': phoneme_surprisal,
        'phoneme_entropy': phoneme_entropy,
        'word_surprisal': word_surprisal,
        'word_entropy': word_entropy,
        'word_onset': word_onset,
        'phoneme_surprisal_shuffled': phoneme_surprisal_shuffled,
        'phoneme_entropy_shuffled': phoneme_entropy_shuffled
    }


def shuffle_predictor(predictor_list):
    """Shuffle non-zero values within each predictor while preserving timing."""
    for i in range(len(predictor_list)):
        nonzero_indices = np.nonzero(predictor_list[i].x)[0]
        nonzeros = predictor_list[i].x[nonzero_indices]
        np.random.shuffle(nonzeros)
        
        for t, idx in enumerate(nonzero_indices):
            predictor_list[i].x[idx] = nonzeros[t]
    
    return predictor_list


# ============================================================================
# Model Definitions
# ============================================================================

def define_models(predictors):
    """Define TRF models with different predictor combinations."""
    p = predictors  # Shorthand
    
    models = {
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+words': 
            [p['gammatone'], p['gammatone_onsets'], p['word_surprisal'], p['word_entropy']],
        
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes+words': 
            [p['gammatone'], p['gammatone_onsets'], p['phoneme_onsets'], 
             p['phoneme_surprisal'], p['phoneme_entropy'], p['word_surprisal'], p['word_entropy']],
        
        'Control2_Delta+Theta_STG_sources_normalized_phonemes+words': 
            [p['gammatone'], p['phoneme_onsets'], p['phoneme_surprisal'], 
             p['phoneme_entropy'], p['word_surprisal'], p['word_entropy']],
        
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+phoneme_surprisal_entropy+words_shuffled_new': 
            [p['gammatone'], p['gammatone_onsets'], p['phoneme_onsets'], 
             p['phoneme_surprisal_shuffled'], p['phoneme_entropy_shuffled'], 
             p['word_surprisal'], p['word_entropy']],
        
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+phoneme_onset+words': 
            [p['gammatone'], p['gammatone_onsets'], p['phoneme_onsets'], 
             p['word_surprisal'], p['word_entropy']],
        
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+phoneme_surp_entp+words': 
            [p['gammatone'], p['gammatone_onsets'], p['phoneme_surprisal'], 
             p['phoneme_entropy'], p['word_surprisal'], p['word_entropy']],
    }
    
    return models


# ============================================================================
# TRF Estimation
# ============================================================================

def estimate_trfs(condition, subject):
    """Estimate TRFs for all models in a given condition."""
    print(f"\n{'='*80}")
    print(f"Processing condition: {condition}, subject: {subject}")
    print(f"{'='*80}\n")
    
    # Setup paths
    meg_dir = DATA_ROOT / 'processed' / subject / 'meg'
    trf_dir = DATA_ROOT / 'processed' / subject / 'meg' / 'TRF' / condition
    trf_dir.mkdir(exist_ok=True)
    
    # Get stimuli
    stimuli = get_stimulus_files(condition)
    print(f"Processing {len(stimuli)} stimuli")
    
    # Load predictors
    print("Loading predictors...")
    predictors = load_predictors(condition, stimuli)
    
    # Define models
    models = define_models(predictors)
    print(f"Defined {len(models)} models")
    
    # Generate TRF paths
    trf_paths_lh = {model: trf_dir / f'{subject} {model}_lh.pickle' for model in models}
    trf_paths_rh = {model: trf_dir / f'{subject} {model}_rh.pickle' for model in models}
    
    # Reconstruct source space
    print("Reconstructing source space...")
    stc_lh_all, stc_rh_all = make_source_space(condition, subject, meg_dir, 
                                                SUBJECTS_DIR, stimuli, 
                                                predictors['gammatone'])
    
    # Concatenate trials
    print("Concatenating trials...")
    stc_lh_concatenated = eelbrain.concatenate(stc_lh_all)
    stc_rh_concatenated = eelbrain.concatenate(stc_rh_all)
    
    # Estimate TRFs for each model
    for model, model_predictors in models.items():
        path_lh = trf_paths_lh[model]
        path_rh = trf_paths_rh[model]
        
        print(f"\nEstimating: {subject} ~ {model}")
        
        # Concatenate predictors
        predictors_concatenated = []
        for predictor in model_predictors:
            predictors_concatenated.append(
                eelbrain.concatenate([predictor[i] for i in range(len(stimuli))])
            )
        
        # Fit left hemisphere TRF
        if not path_lh.exists():
            print("  Fitting left hemisphere...")
            trf_lh = eelbrain.boosting(
                stc_lh_concatenated, predictors_concatenated, 
                TMIN, TMAX, error='l2', basis=BASIS, 
                partitions=PARTITIONS, test=TEST, selective_stopping=True
            )
            eelbrain.save.pickle(trf_lh, path_lh)
            del trf_lh
        else:
            print("  Left hemisphere TRF already exists")
        
        # Fit right hemisphere TRF
        if not path_rh.exists():
            print("  Fitting right hemisphere...")
            trf_rh = eelbrain.boosting(
                stc_rh_concatenated, predictors_concatenated, 
                TMIN, TMAX, error='l2', basis=BASIS, 
                partitions=PARTITIONS, test=TEST, selective_stopping=True
            )
            eelbrain.save.pickle(trf_rh, path_rh)
            del trf_rh
        else:
            print("  Right hemisphere TRF already exists")
    
    print(f"\nCompleted processing for condition: {condition}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("TRF Estimation Pipeline")
    print("="*80)
    print(f"Subject: {SUBJECT}")
    print(f"Region: {REGION}")
    print(f"Conditions: {CONDITIONS}")
    print("="*80)
    
    for condition in CONDITIONS:
        estimate_trfs(condition, SUBJECT)
    
    print("\n" + "="*80)
    print("All processing completed!")
    print("="*80)


if __name__ == "__main__":
    main()