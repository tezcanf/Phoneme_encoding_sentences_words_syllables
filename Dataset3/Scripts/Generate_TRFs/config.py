"""
Configuration module for TRF estimation pipeline.
Contains all paths, constants, and condition mappings.
"""
from pathlib import Path
from typing import Dict, List

# ============================================================================
# Directory Paths
# ============================================================================

DATA_ROOT = Path.cwd().parents[1]
STIMULUS_BASE_DIR = DATA_ROOT / 'Materials' / 'Stimuli' / 'Sounds_Syllables'
PREDICTOR_BASE_DIR = DATA_ROOT  / 'Materials' / 'Predictors'
PATH_BLOCK = DATA_ROOT / 'Materials' / 'Stimuli' / 'Block_order'
TRF_BASE_DIR = DATA_ROOT  / 'TRF_models'

# ============================================================================
# Subject List
# ============================================================================
SUBJECTS = [
    'sub-006', 'sub-008', 'sub-009', 'sub-010', 'sub-011', 'sub-012',
    'sub-013', 'sub-015', 'sub-016', 'sub-017', 'sub-018', 'sub-019',
    'sub-020', 'sub-021', 'sub-022', 'sub-023', 'sub-024', 'sub-025',
    'sub-026', 'sub-027', 'sub-028', 'sub-029', 'sub-030', 'sub-031',
    'sub-032', 'sub-033', 'sub-034', 'sub-035', 'sub-036', 'sub-037',
    'sub-038',
]

# ============================================================================
# Condition Definitions
# ============================================================================
CONDITIONS = [
    'sentences',      # Natural sentences (formerly 'control_sentence')
    'words',          # Random word lists (formerly 'random_word_list')
    'syllables',      # Random syllables (formerly 'random_syllables')
]

# Mapping from new simplified names to original directory/file names
CONDITION_DIR_MAPPING = {
    'sentences': 'control_sentence',
    'words': 'random_word_list',
    'syllables': 'random_syllables',
}

# Event ID mapping for MEG data
EVENT_DICT_CONDITIONS: Dict[str, int] = {
    'controlsentence': 10,
    'semanticallyanomalous': 20,
    'syntacticallyanomalous': 30,
    'lexicosemanticgrouping': 40,  # old name for random_word_list
    'randomwordlist': 40,
    'fixedwordorders': 50,
    'randomsyllables': 70,
}

# Mapping from simplified condition names to epoch names (for backward compatibility)
CONDITION_TO_EPOCH_NAME: Dict[str, str] = {
    'sentences': 'controlsentence',
    'words': 'lexicosemanticgrouping',  # uses old epoch name
    'syllables': 'randomsyllables',
}

# Legacy condition names for block order files
BLOCK_ORDER_CONDITION_NAMES: Dict[str, str] = {
    'words': 'lexico_semantic_grouping',  # block order files use this name
}

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
}

# Predictor parameters
PREDICTOR_PARAMS = {
    'resampling_rate': 0.01,  # 100 Hz
    'pad_tstart': -0.05,
    'pad_tstop_offset': 0.7,
}

# Source localization parameters
SOURCE_PARAMS = {
    'crop_tmin': 1.55,
    'crop_tmax': 27.2,  # will add tmax to this
    'filter_highpass': 8.0,
    'resample_freq': 100,
    'region': 'STG',  # Fixed to Superior Temporal Gyrus
}

# ============================================================================
# File Name Templates
# ============================================================================
EPOCHS_FILENAME = '{subject}_resampled_300Hz-05-145Hz_filtered-ICA-eyeblink-epochs.fif'
EVENTS_FILENAME = '{subject}-eve.fif'
FORWARD_FILENAME = '{subject}_forward_ICA_all.fif'
INVERSE_FILENAME = '{subject}_inverse_ICA_all.fif'
TRANS_FILENAME = '{subject}-trans.fif'

# ============================================================================
# Helper Functions
# ============================================================================
def get_subject_paths(subject: str) -> Dict[str, Path]:
    """
    Get all relevant paths for a given subject.
    
    Args:
        subject: Subject ID (e.g., 'sub-006')
        
    Returns:
        Dictionary containing all relevant paths
    """
    meg_dir = DATA_ROOT / 'processed' / subject / 'meg'
    subjects_dir = DATA_ROOT / 'processed'
    
    return {
        'meg_dir': meg_dir,
        'subjects_dir': subjects_dir,
        'epochs': meg_dir / EPOCHS_FILENAME.format(subject=subject),
        'events': meg_dir / EVENTS_FILENAME.format(subject=subject),
        'forward': meg_dir / FORWARD_FILENAME.format(subject=subject),
        'inverse': meg_dir / INVERSE_FILENAME.format(subject=subject),
        'trans': subjects_dir / subject / 'meg' / TRANS_FILENAME.format(subject=subject),
    }

def get_condition_paths(condition: str) -> Dict[str, Path]:
    """
    Get stimulus and predictor directories for a given condition.
    
    Args:
        condition: Condition name ('sentences', 'words', or 'syllables')
        
    Returns:
        Dictionary containing stimulus and predictor directories
    """
    # Map simplified name to actual directory name
    dir_name = CONDITION_DIR_MAPPING.get(condition, condition)
    
    return {
        'stimulus_dir': STIMULUS_BASE_DIR / dir_name,
        'predictor_dir': PREDICTOR_BASE_DIR / dir_name,
    }

def get_trf_output_dir(condition: str, subject: str) -> Path:
    """
    Get output directory for TRF models.
    
    Args:
        condition: Condition name ('sentences', 'words', or 'syllables')
        subject: Subject ID
        
    Returns:
        Path to TRF output directory
    """
    # Map simplified name to actual directory name
    dir_name = CONDITION_DIR_MAPPING.get(condition, condition)
    
    trf_dir = TRF_BASE_DIR / dir_name / subject
    trf_dir.mkdir(parents=True, exist_ok=True)
    return trf_dir
