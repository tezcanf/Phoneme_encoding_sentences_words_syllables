"""
Predictor loading and preprocessing utilities.
Handles gammatone spectrograms, phoneme features, and word features.
"""
from pathlib import Path
from typing import List, Dict, Any

import eelbrain
import trftools

from config import PREDICTOR_PARAMS


def load_gammatone_spectrograms(
    predictor_dir: Path,
    stimuli: List[str]
) -> List[Any]:
    """
    Load and preprocess gammatone spectrograms.
    
    Args:
        predictor_dir: Directory containing predictor files
        stimuli: List of stimulus names
        
    Returns:
        List of preprocessed gammatone spectrograms
    """
    # Load spectrograms
    gammatone = [
        eelbrain.load.unpickle(predictor_dir / f'{stimulus}~gammatone-8.pickle')
        for stimulus in stimuli
    ]
    
    # Resample to 100 Hz
    gammatone = [
        x.bin(PREDICTOR_PARAMS['resampling_rate'], dim='time', label='start')
        for x in gammatone
    ]
    
    # Pad onset and offset
    gammatone = [
        trftools.pad(
            x,
            tstart=PREDICTOR_PARAMS['pad_tstart'],
            tstop=x.time.tstop + PREDICTOR_PARAMS['pad_tstop_offset'],
            name='gammatone'
        )
        for x in gammatone
    ]
    
    return gammatone


def load_gammatone_onsets(
    predictor_dir: Path,
    stimuli: List[str],
    gammatone: List[Any]
) -> List[Any]:
    """
    Load and preprocess gammatone onset spectrograms.
    
    Args:
        predictor_dir: Directory containing predictor files
        stimuli: List of stimulus names
        gammatone: Reference gammatone spectrograms for time alignment
        
    Returns:
        List of preprocessed gammatone onset spectrograms
    """
    # Load onset spectrograms
    gammatone_onsets = [
        eelbrain.load.unpickle(predictor_dir / f'{stimulus}~gammatone-on-8.pickle')
        for stimulus in stimuli
    ]
    
    # Resample
    gammatone_onsets = [
        x.bin(PREDICTOR_PARAMS['resampling_rate'], dim='time', label='start')
        for x in gammatone_onsets
    ]
    
    # Align time dimension with gammatone spectrograms
    gammatone_onsets = [
        eelbrain.set_time(x, gt.time, name='gammatone_on')
        for x, gt in zip(gammatone_onsets, gammatone)
    ]
    
    return gammatone_onsets


def load_phoneme_predictors(
    predictor_dir: Path,
    stimuli: List[str],
    gammatone: List[Any]
) -> Dict[str, List[Any]]:
    """
    Load phoneme-level predictors (onsets, surprisal, entropy).
    
    Args:
        predictor_dir: Directory containing predictor files
        stimuli: List of stimulus names
        gammatone: Reference gammatone spectrograms for time alignment
        
    Returns:
        Dictionary containing phoneme predictors
    """
    # Load phoneme tables
    phoneme_tables = [
        eelbrain.load.unpickle(predictor_dir / f'{stimulus}~phoneme_cohort_model.pickle')
        for stimulus in stimuli
    ]
    
    # Create impulse predictors
    phoneme_onsets = [
        eelbrain.event_impulse_predictor(gt.time, ds=ds, name='phonemes')
        for gt, ds in zip(gammatone, phoneme_tables)
    ]
    
    phoneme_surprisal = [
        eelbrain.event_impulse_predictor(
            gt.time, value='cohort_surprisal', ds=ds, name='cohort_surprisal'
        )
        for gt, ds in zip(gammatone, phoneme_tables)
    ]
    
    phoneme_entropy = [
        eelbrain.event_impulse_predictor(
            gt.time, value='cohort_entropy', ds=ds, name='cohort_entropy'
        )
        for gt, ds in zip(gammatone, phoneme_tables)
    ]
    
    return {
        'phoneme_onsets': phoneme_onsets,
        'phoneme_surprisal': phoneme_surprisal,
        'phoneme_entropy': phoneme_entropy,
    }


def load_word_predictors(
    predictor_dir: Path,
    stimuli: List[str],
    gammatone: List[Any]
) -> Dict[str, List[Any]]:
    """
    Load word-level predictors (surprisal, entropy, onsets).
    
    Args:
        predictor_dir: Directory containing predictor files
        stimuli: List of stimulus names
        gammatone: Reference gammatone spectrograms for time alignment
        
    Returns:
        Dictionary containing word predictors
    """
    # Load word tables
    phoneme_tables_GPT = [
        eelbrain.load.unpickle(predictor_dir / f'{stimulus}~phoneme_cohort_model_GPT.pickle')
        for stimulus in stimuli
    ]
    
    # Create impulse predictors
    word_surprisal = [
        eelbrain.event_impulse_predictor(
            gt.time, value='word_surprisal_GPT', ds=ds, name='word_surprisal'
        )
        for gt, ds in zip(gammatone, phoneme_tables_GPT)
    ]
    
    word_entropy = [
        eelbrain.event_impulse_predictor(
            gt.time, value='word_entropy_GPT', ds=ds, name='word_entropy'
        )
        for gt, ds in zip(gammatone, phoneme_tables_GPT)
    ]
    
    word_onset = [
        eelbrain.event_impulse_predictor(
            gt.time, value='word_number', ds=ds, name='word_number'
        )
        for gt, ds in zip(gammatone, phoneme_tables_GPT)
    ]
    
    return {
        'word_surprisal': word_surprisal,
        'word_entropy': word_entropy,
        'word_onset': word_onset,
    }


def load_all_predictors(
    predictor_dir: Path,
    stimuli: List[str],
    include_phonemes: bool = True,
    include_words: bool = True
) -> Dict[str, List[Any]]:
    """
    Load all predictors for a set of stimuli.
    
    Args:
        predictor_dir: Directory containing predictor files
        stimuli: List of stimulus names
        include_phonemes: Whether to load phoneme predictors
        include_words: Whether to load word predictors
        
    Returns:
        Dictionary containing all predictors
    """
    predictors = {}
    
    # Load acoustic predictors (always included)
    gammatone = load_gammatone_spectrograms(predictor_dir, stimuli)
    gammatone_onsets = load_gammatone_onsets(predictor_dir, stimuli, gammatone)
    
    predictors['gammatone'] = gammatone
    predictors['gammatone_onsets'] = gammatone_onsets
    
    # Load phoneme predictors if requested
    if include_phonemes:
        phoneme_preds = load_phoneme_predictors(predictor_dir, stimuli, gammatone)
        predictors.update(phoneme_preds)
    
    # Load word predictors if requested
    if include_words:
        word_preds = load_word_predictors(predictor_dir, stimuli, gammatone)
        predictors.update(word_preds)
    
    return predictors


def get_nsamples(predictors: Dict[str, List[Any]]) -> int:
    """
    Get number of time samples from predictors.
    
    Args:
        predictors: Dictionary of predictors
        
    Returns:
        Number of time samples
    """
    gammatone = predictors['gammatone']
    return gammatone[0].time.nsamples
