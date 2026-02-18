"""
Configuration file for TRF estimation

This module centralizes all paths and parameters for TRF estimation.
It handles both Dutch and Chinese participants with Dutch and Chinese stimuli.
"""
from pathlib import Path

# ============================================================================
# Subject Lists
# ============================================================================

DUTCH_SUBJECTS = [
    'sub-003', 'sub-005', 'sub-007', 'sub-008', 'sub-009',
    'sub-010', 'sub-012', 'sub-013', 'sub-014', 'sub-015',
    'sub-016', 'sub-017', 'sub-018', 'sub-019', 'sub-020'
]

CHINESE_SUBJECTS = [
    'sub-021', 'sub-022', 'sub-023', 'sub-024', 'sub-025',
    'sub-026', 'sub-027', 'sub-028', 'sub-029', 'sub-030',
    'sub-032', 'sub-033', 'sub-034', 'sub-035'
]

# ============================================================================
# Experimental Conditions
# ============================================================================

CONDITIONS = ['Words', 'Syllables']

# ============================================================================
# Root Directory Configuration
# ============================================================================

# Get the root directory (two levels up from current working directory)
ROOT_DIR = Path.cwd().parents[1]

# ============================================================================
# Path Construction Functions
# ============================================================================

def get_materials_root(stimulus_type):
    """
    Get the Materials root directory for a given stimulus type
    
    Parameters
    ----------
    stimulus_type : str
        'Dutch' or 'Chinese'
    
    Returns
    -------
    Path
        Path to Materials directory for the stimulus type
    """
    if stimulus_type not in ['Dutch', 'Chinese']:
        raise ValueError(f"stimulus_type must be 'Dutch' or 'Chinese', got {stimulus_type}")
    
    return ROOT_DIR / 'Materials' / f'{stimulus_type}_stimuli'


def get_meg_root(subject_group):
    """
    Get the MEG data root directory for a subject group
    
    Parameters
    ----------
    subject_group : str
        'Dutch' or 'Chinese'
    
    Returns
    -------
    Path
        Path to MEG root directory
    """
    if subject_group not in ['Dutch', 'Chinese']:
        raise ValueError(f"subject_group must be 'Dutch' or 'Chinese', got {subject_group}")
    
    return ROOT_DIR / f'{subject_group}_participants' / 'processed'


# ============================================================================
# TRF Estimation Parameters
# ============================================================================

TRF_PARAMS = {
    'tmin': -0.05,
    'tmax': 0.7,
    'error': 'l2',
    'basis': 0.050,
    'partitions': 5,
    'test': 1,
    'selective_stopping': True,
    'n_stimuli': 40
}

# ============================================================================
# Model Definitions
# ============================================================================

def get_models(stimulus_type='Dutch'):
    """
    Get model configurations based on stimulus type
    
    Parameters
    ----------
    stimulus_type : str
        'Dutch' or 'Chinese' to determine naming convention
    
    Returns
    -------
    dict
        Dictionary mapping model names to lists of predictor names
    """
    if stimulus_type not in ['Dutch', 'Chinese']:
        raise ValueError(f"stimulus_type must be 'Dutch' or 'Chinese', got {stimulus_type}")
    
    prefix = f'Control2_Delta+Theta_STG_sources_normalized_{stimulus_type}_stimuli'
    
    return {
        f'{prefix}_acoustic': ['gammatone', 'gammatone_onsets'],
        f'{prefix}_acoustic+phonemes': ['gammatone', 'gammatone_onsets', 'phoneme_onsets', 
                                         'phoneme_surprisal', 'phoneme_entropy'],
        f'{prefix}_phonemes': ['gammatone', 'phoneme_onsets', 'phoneme_surprisal', 'phoneme_entropy'],
        f'{prefix}_phoneme_surp_ent': ['gammatone', 'gammatone_onsets', 'phoneme_surprisal', 
                                        'phoneme_entropy'],
        f'{prefix}_phoneme_onset': ['gammatone', 'gammatone_onsets', 'phoneme_onsets'],
    }


# ============================================================================
# Subject Configuration
# ============================================================================

def get_subject_config(subject, stimulus_type='native'):
    """
    Get configuration for a specific subject and stimulus type
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-003')
    stimulus_type : str
        'native', 'non-native', 'Dutch', or 'Chinese'
        - 'native': Dutch subjects get Dutch stimuli, Chinese subjects get Chinese stimuli
        - 'non-native': Dutch subjects get Chinese stimuli, Chinese subjects get Dutch stimuli
        - 'Dutch': All subjects get Dutch stimuli
        - 'Chinese': All subjects get Chinese stimuli
    
    Returns
    -------
    dict
        Configuration dictionary with paths and parameters
        Keys: subject_group, stimulus_type, participant_folder, 
              stimuli_dir, predictors_dir, meg_root, subjects_dir
    """
    # Validate subject
    if subject in DUTCH_SUBJECTS:
        subject_group = 'Dutch'
    elif subject in CHINESE_SUBJECTS:
        subject_group = 'Chinese'
    else:
        raise ValueError(f"Unknown subject: {subject}. Must be in DUTCH_SUBJECTS or CHINESE_SUBJECTS")
    
    # Determine actual stimulus type based on subject and parameter
    if stimulus_type == 'native':
        actual_stim = 'Dutch' if subject_group == 'Dutch' else 'Chinese'
    elif stimulus_type == 'non-native':
        actual_stim = 'Chinese' if subject_group == 'Dutch' else 'Dutch'
    elif stimulus_type in ['Dutch', 'Chinese']:
        actual_stim = stimulus_type
    else:
        raise ValueError(f"Unknown stimulus_type: {stimulus_type}. "
                        f"Must be 'native', 'non-native', 'Dutch', or 'Chinese'")
    
    # Construct paths
    participant_folder = f'{subject_group}_participants'
    materials_root = get_materials_root(actual_stim)
    meg_root = get_meg_root(subject_group)
    
    return {
        'subject_group': subject_group,
        'stimulus_type': actual_stim,
        'participant_folder': participant_folder,
        'stimuli_dir': materials_root / 'Stimuli' / participant_folder,
        'predictors_dir': materials_root / 'Predictors' / participant_folder,
        'meg_root': meg_root,
        'subjects_dir': meg_root
    }


# ============================================================================
# Validation Helper
# ============================================================================

def validate_paths(config, subject):
    """
    Validate that key paths exist for a given configuration
    
    Parameters
    ----------
    config : dict
        Configuration dictionary from get_subject_config
    subject : str
        Subject ID
    
    Returns
    -------
    dict
        Dictionary with validation results
    """
    results = {
        'stimuli_dir': config['stimuli_dir'].exists(),
        'predictors_dir': config['predictors_dir'].exists(),
        'meg_dir': (config['meg_root'] / subject / 'meg').exists(),
        'subjects_dir': (config['subjects_dir'] / subject).exists()
    }
    
    return results


if __name__ == '__main__':
    # Print example configuration
    print("=" * 80)
    print("CONFIGURATION EXAMPLES")
    print("=" * 80)
    
    print("\nExample 1: Dutch subject with native (Dutch) stimuli")
    config = get_subject_config('sub-003', 'native')
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nExample 2: Chinese subject with native (Chinese) stimuli")
    config = get_subject_config('sub-021', 'native')
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nExample 3: Dutch subject with non-native (Chinese) stimuli")
    config = get_subject_config('sub-003', 'non-native')
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
