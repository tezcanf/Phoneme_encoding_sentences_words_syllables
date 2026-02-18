#!/usr/bin/env python3
"""
Interactive TRF Estimation Runner

This script can be run in Jupyter notebooks or IPython without command-line arguments.
Just set the variables in the configuration section and run the script.

This is useful for:
- Testing with a single subject
- Running in Jupyter notebooks
- Quick interactive exploration
"""

import os
from pathlib import Path
import eelbrain
import trftools
from config import (
    DUTCH_SUBJECTS, CHINESE_SUBJECTS, CONDITIONS, TRF_PARAMS,
    get_models, get_subject_config
)
from utils_unified import make_source_space

# ============================================================================
# CONFIGURATION - Edit these variables to control what gets processed
# ============================================================================

# Choose subjects to process
# Options:
#   - List specific subjects: ['sub-003', 'sub-005']
#   - All Dutch subjects: DUTCH_SUBJECTS
#   - All Chinese subjects: CHINESE_SUBJECTS
#   - All subjects: DUTCH_SUBJECTS + CHINESE_SUBJECTS
SUBJECTS_TO_PROCESS = ['sub-003']  # <-- EDIT THIS

# Choose conditions to process
# Options: ['Words'], ['Syllables'], or ['Words', 'Syllables']
CONDITIONS_TO_PROCESS = ['Words']  # <-- EDIT THIS

# Choose stimulus type
# Options:
#   - 'native': Dutch subjects get Dutch stimuli, Chinese subjects get Chinese stimuli
#   - 'non-native': Dutch subjects get Chinese stimuli, Chinese subjects get Dutch stimuli
#   - 'Dutch': All subjects get Dutch stimuli
#   - 'Chinese': All subjects get Chinese stimuli
STIMULUS_TYPE = 'native'  # <-- EDIT THIS

# ============================================================================
# Helper Functions
# ============================================================================

def load_predictors(predictor_dir, stimuli, params):
    """Load and prepare all predictors for TRF estimation"""
    print("  Loading predictors...")
    
    # Load gammatone spectrograms
    print("    Loading gammatone spectrograms...")
    gammatone = [eelbrain.load.unpickle(predictor_dir / f'{stimulus}~gammatone-8.pickle') 
                 for stimulus in stimuli]
    gammatone = [x.bin(0.01, dim='time', label='start') for x in gammatone]
    gammatone = [trftools.pad(x, tstart=params['tmin'], 
                              tstop=x.time.tstop + params['tmax'], 
                              name='gammatone') 
                 for x in gammatone]
    
    # Load onset spectrograms
    print("    Loading onset spectrograms...")
    gammatone_onsets = [eelbrain.load.unpickle(predictor_dir / f'{stimulus}~gammatone-on-8.pickle') 
                        for stimulus in stimuli]
    gammatone_onsets = [x.bin(0.01, dim='time', label='start') for x in gammatone_onsets]
    gammatone_onsets = [eelbrain.set_time(x, gt.time, name='gammatone_on') 
                        for x, gt in zip(gammatone_onsets, gammatone)]
    
    # Load phoneme tables
    print("    Loading phoneme tables...")
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
    
    print(f"    ✓ Loaded {len(stimuli)} predictor sets")
    
    return {
        'gammatone': gammatone,
        'gammatone_onsets': gammatone_onsets,
        'phoneme_onsets': phoneme_onsets,
        'phoneme_surprisal': phoneme_surprisal,
        'phoneme_entropy': phoneme_entropy
    }


def get_stimuli_list(stimulus_dir, subject, n_stimuli=40):
    """Get list of stimuli for a given subject"""
    if not stimulus_dir.exists():
        raise FileNotFoundError(f"Stimulus directory not found: {stimulus_dir}")
    
    stimuli = [f.split('.')[0][:-1] for f in os.listdir(stimulus_dir) 
               if (subject in f) and (f.endswith('.wav'))]
    stimuli.sort()
    
    if len(stimuli) == 0:
        raise ValueError(f"No stimulus files found for {subject} in {stimulus_dir}")
    
    return stimuli[:n_stimuli]


def estimate_trfs_for_subject(subject, condition, config, models, params):
    """Estimate TRFs for a single subject and condition"""
    print(f"\n{'='*80}")
    print(f"Processing {subject} - {condition} - {config['stimulus_type']} stimuli")
    print(f"{'='*80}\n")
    
    # Set up paths using config
    stimulus_dir = config['stimuli_dir'] / condition
    predictor_dir = config['predictors_dir'] / condition
    meg_dir = config['meg_root'] / subject / 'meg'
    trf_dir = meg_dir / 'TRF' / condition
    
    # Create output directory
    trf_dir.mkdir(exist_ok=True, parents=True)
    print(f"  Output directory: {trf_dir}")
    
    # Get stimuli list
    print(f"\n  Loading stimulus list...")
    stimuli = get_stimuli_list(stimulus_dir, subject, params['n_stimuli'])
    print(f"  ✓ Found {len(stimuli)} stimuli for {subject}")
    
    # Load predictors
    print(f"\n  Loading predictors from: {predictor_dir}")
    all_predictors = load_predictors(predictor_dir, stimuli, params)
    
    # Load source space data
    print(f"\n  Loading source space data...")
    print(f"    MEG directory: {meg_dir}")
    print(f"    Subjects directory: {config['subjects_dir']}")
    
    stc_lh_all, stc_rh_all = make_source_space(
        condition, subject, meg_dir, str(config['subjects_dir']), stimuli,
        stimulus_type=config['stimulus_type']
    )
    print(f"  ✓ Loaded {len(stc_lh_all)} source space trials")
    
    # Concatenate trials
    print(f"\n  Concatenating trials...")
    stc_lh_ndvar_concatenated = eelbrain.concatenate(stc_lh_all)
    stc_rh_ndvar_concatenated = eelbrain.concatenate(stc_rh_all)
    print(f"  ✓ Concatenated data shape: {stc_lh_ndvar_concatenated.shape}")
    
    # Estimate TRFs for each model
    n_models = len(models)
    for idx, (model_name, predictor_names) in enumerate(models.items(), 1):
        print(f"\n{'-'*80}")
        print(f"Model {idx}/{n_models}: {model_name}")
        print(f"  Predictors: {', '.join(predictor_names)}")
        print(f"{'-'*80}")
        
        # Set up output paths
        path_lh = trf_dir / f'{subject} {model_name}_lh.pickle'
        path_rh = trf_dir / f'{subject} {model_name}_rh.pickle'
        
        # Check if files already exist
        if path_lh.exists() and path_rh.exists():
            print(f"  ✓ TRFs already exist, skipping...")
            continue
        
        # Prepare predictors for this model
        print(f"  Preparing predictors...")
        predictors_concatenated = []
        for pred_name in predictor_names:
            predictor_list = all_predictors[pred_name]
            concatenated = eelbrain.concatenate([predictor_list[i] for i in range(len(stimuli))])
            predictors_concatenated.append(concatenated)
        print(f"  ✓ Prepared {len(predictors_concatenated)} predictors")
        
        # Estimate left hemisphere TRF
        if not path_lh.exists():
            print(f"  Estimating left hemisphere TRF...")
            trf_lh = eelbrain.boosting(
                stc_lh_ndvar_concatenated, 
                predictors_concatenated,
                params['tmin'], 
                params['tmax'],
                error=params['error'],
                basis=params['basis'],
                partitions=params['partitions'],
                test=params['test'],
                selective_stopping=params['selective_stopping']
            )
            eelbrain.save.pickle(trf_lh, path_lh)
            print(f"  ✓ Saved: {path_lh.name}")
            del trf_lh
        else:
            print(f"  ✓ Left hemisphere TRF already exists")
        
        # Estimate right hemisphere TRF
        if not path_rh.exists():
            print(f"  Estimating right hemisphere TRF...")
            trf_rh = eelbrain.boosting(
                stc_rh_ndvar_concatenated,
                predictors_concatenated,
                params['tmin'],
                params['tmax'],
                error=params['error'],
                basis=params['basis'],
                partitions=params['partitions'],
                test=params['test'],
                selective_stopping=params['selective_stopping']
            )
            eelbrain.save.pickle(trf_rh, path_rh)
            print(f"  ✓ Saved: {path_rh.name}")
            del trf_rh
        else:
            print(f"  ✓ Right hemisphere TRF already exists")
    
    print(f"\n✓ Completed {subject} - {condition}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("TRF ESTIMATION - INTERACTIVE MODE")
    print("="*80)
    print(f"\nSubjects: {SUBJECTS_TO_PROCESS}")
    print(f"Conditions: {CONDITIONS_TO_PROCESS}")
    print(f"Stimulus type: {STIMULUS_TYPE}")
    print()
    
    # Process each subject and condition
    n_completed = 0
    n_errors = 0
    
    for subject in SUBJECTS_TO_PROCESS:
        for condition in CONDITIONS_TO_PROCESS:
            try:
                # Get configuration based on subject and stimulus type
                config = get_subject_config(subject, STIMULUS_TYPE)
                
                # Get appropriate models
                models = get_models(config['stimulus_type'])
                
                # Estimate TRFs
                estimate_trfs_for_subject(subject, condition, config, models, TRF_PARAMS)
                
                n_completed += 1
                
            except Exception as e:
                print(f"\n✗ Error processing {subject} - {condition}: {e}")
                import traceback
                traceback.print_exc()
                n_errors += 1
                continue
    
    # Print summary
    print("\n" + "="*80)
    print("TRF ESTIMATION SUMMARY")
    print("="*80)
    print(f"Successfully completed: {n_completed}")
    print(f"Errors: {n_errors}")
    
    if n_errors == 0:
        print("\n✓ All TRF estimations completed successfully!")
    else:
        print(f"\n⚠ Completed with {n_errors} error(s)")
    
    print("="*80)
