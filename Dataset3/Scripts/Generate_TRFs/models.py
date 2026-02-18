"""
TRF model definitions.
Specifies different combinations of predictors for TRF estimation.
"""
from typing import Dict, List, Any


def get_models_with_words(predictors: Dict[str, List[Any]]) -> Dict[str, List[List[Any]]]:
    """
    Define TRF models that include word-level predictors.
    
    Used for: sentences condition
    
    Args:
        predictors: Dictionary of all available predictors
        
    Returns:
        Dictionary mapping model names to lists of predictors
    """
    models = {
        # Acoustic + words
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+words': [
            predictors['gammatone'],
            predictors['gammatone_onsets'],
            predictors['word_surprisal'],
            predictors['word_entropy'],
        ],
        
        # Acoustic + phonemes + words (full model)
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes+words': [
            predictors['gammatone'],
            predictors['gammatone_onsets'],
            predictors['phoneme_onsets'],
            predictors['phoneme_surprisal'],
            predictors['phoneme_entropy'],
            predictors['word_surprisal'],
            predictors['word_entropy'],
        ],
        
        # Phonemes + words (no acoustic)
        'Control2_Delta+Theta_STG_sources_normalized_phonemes+words': [
            predictors['gammatone'],
            predictors['phoneme_onsets'],
            predictors['phoneme_surprisal'],
            predictors['phoneme_entropy'],
            predictors['word_surprisal'],
            predictors['word_entropy'],
        ],
        
        # Acoustic + phoneme onsets + words
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+phoneme_onset+words': [
            predictors['gammatone'],
            predictors['gammatone_onsets'],
            predictors['phoneme_onsets'],
            predictors['word_surprisal'],
            predictors['word_entropy'],
        ],
        
        # Acoustic + phoneme surprisal/entropy + words
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+phoneme_surp_entp+words': [
            predictors['gammatone'],
            predictors['gammatone_onsets'],
            predictors['phoneme_surprisal'],
            predictors['phoneme_entropy'],
            predictors['word_surprisal'],
            predictors['word_entropy'],
        ],
    }
    
    return models


def get_models_with_phonemes_only(predictors: Dict[str, List[Any]]) -> Dict[str, List[List[Any]]]:
    """
    Define TRF models with phoneme-level predictors only (no words).
    
    Used for: syllables condition
    
    Args:
        predictors: Dictionary of all available predictors
        
    Returns:
        Dictionary mapping model names to lists of predictors
    """
    models = {
        # Acoustic only
        'Control2_Delta+Theta_STG_sources_normalized_acoustic': [
            predictors['gammatone'],
            predictors['gammatone_onsets'],
        ],
        
        # Acoustic + all phoneme features
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes': [
            predictors['gammatone'],
            predictors['gammatone_onsets'],
            predictors['phoneme_onsets'],
            predictors['phoneme_surprisal'],
            predictors['phoneme_entropy'],
        ],
        
        # Phonemes only (no acoustic)
        'Control2_Delta+Theta_STG_sources_normalized_phonemes': [
            predictors['gammatone'],
            predictors['phoneme_onsets'],
            predictors['phoneme_surprisal'],
            predictors['phoneme_entropy'],
        ],
        
        # Acoustic + phoneme onsets only
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+phoneme_onset': [
            predictors['gammatone'],
            predictors['gammatone_onsets'],
            predictors['phoneme_onsets'],
        ],
        
        # Acoustic + phoneme surprisal/entropy
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+phoneme_surp_entp': [
            predictors['gammatone'],
            predictors['gammatone_onsets'],
            predictors['phoneme_surprisal'],
            predictors['phoneme_entropy'],
        ],
    }
    
    return models


def get_models_full_set(predictors: Dict[str, List[Any]]) -> Dict[str, List[List[Any]]]:
    """
    Define comprehensive set of TRF models including both phoneme and word predictors.
    
    Used for: words condition
    
    Args:
        predictors: Dictionary of all available predictors
        
    Returns:
        Dictionary mapping model names to lists of predictors
    """
    # Start with phoneme-only models
    models = get_models_with_phonemes_only(predictors)
    
    # Add word-level models
    models.update({
        # Acoustic + words
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+words': [
            predictors['gammatone'],
            predictors['gammatone_onsets'],
            predictors['word_surprisal'],
            predictors['word_entropy'],
        ],
        
        # Acoustic + phonemes + words
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes+words': [
            predictors['gammatone'],
            predictors['gammatone_onsets'],
            predictors['phoneme_onsets'],
            predictors['phoneme_surprisal'],
            predictors['phoneme_entropy'],
            predictors['word_surprisal'],
            predictors['word_entropy'],
        ],
        
        # Phonemes + words
        'Control2_Delta+Theta_STG_sources_normalized_phonemes+words': [
            predictors['gammatone'],
            predictors['phoneme_onsets'],
            predictors['phoneme_surprisal'],
            predictors['phoneme_entropy'],
            predictors['word_surprisal'],
            predictors['word_entropy'],
        ],
        
        # Acoustic + phoneme onsets + words
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+phoneme_onset+words': [
            predictors['gammatone'],
            predictors['gammatone_onsets'],
            predictors['phoneme_onsets'],
            predictors['word_surprisal'],
            predictors['word_entropy'],
        ],
        
        # Acoustic + phoneme surprisal/entropy + words
        'Control2_Delta+Theta_STG_sources_normalized_acoustic+phoneme_surp_entp+words': [
            predictors['gammatone'],
            predictors['gammatone_onsets'],
            predictors['phoneme_surprisal'],
            predictors['phoneme_entropy'],
            predictors['word_surprisal'],
            predictors['word_entropy'],
        ],
    })
    
    return models


def get_models_for_condition(
    condition: str,
    predictors: Dict[str, List[Any]]
) -> Dict[str, List[List[Any]]]:
    """
    Get appropriate TRF models based on the condition.
    
    Args:
        condition: Experimental condition name ('sentences', 'words', or 'syllables')
        predictors: Dictionary of all available predictors
        
    Returns:
        Dictionary mapping model names to lists of predictors
    """
    # Sentences: word-level information
    if condition == 'sentences':
        return get_models_with_words(predictors)
    
    # Words: full model set (both phoneme and word predictors)
    elif condition == 'words':
        return get_models_full_set(predictors)
    
    # Syllables: phoneme-level only, no words
    elif condition == 'syllables':
        return get_models_with_phonemes_only(predictors)
    
    else:
        # Default to full set for unknown conditions
        print(f"Warning: Unknown condition '{condition}'. Using full model set.")
        return get_models_full_set(predictors)
