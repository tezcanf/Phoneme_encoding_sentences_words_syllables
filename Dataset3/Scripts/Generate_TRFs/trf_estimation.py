"""
TRF estimation utilities.
"""
from pathlib import Path
from typing import Dict, List, Any

import eelbrain

from config import TRF_PARAMS


def estimate_trfs(
    stc_lh_concatenated: Any,
    stc_rh_concatenated: Any,
    models: Dict[str, List[List[Any]]],
    stimuli_subject: List[str],
    trf_output_dir: Path,
    subject: str,
    force_recompute: bool = False
) -> None:
    """
    Estimate TRFs for all models and save to disk.
    
    Args:
        stc_lh_concatenated: Concatenated left hemisphere source data
        stc_rh_concatenated: Concatenated right hemisphere source data
        models: Dictionary mapping model names to predictor lists
        stimuli_subject: List of stimulus names (for indexing predictors)
        trf_output_dir: Directory to save TRF models
        subject: Subject ID
        force_recompute: If True, recompute even if file exists
    """
    # Generate TRF file paths
    trf_paths_lh = {
        model: trf_output_dir / f'{subject} {model}_lh.pickle'
        for model in models
    }
    trf_paths_rh = {
        model: trf_output_dir / f'{subject} {model}_rh.pickle'
        for model in models
    }
    
    # Estimate each model
    for model, predictors in models.items():
        path_lh = trf_paths_lh[model]
        path_rh = trf_paths_rh[model]
        
        # Check if already computed
        if not force_recompute and path_lh.exists() and path_rh.exists():
            print(f"Skipping {subject} ~ {model} (already computed)")
            continue
        
        print(f"Estimating: {subject} ~ {model}")
        
        # Concatenate predictors
        predictors_concatenated = _concatenate_predictors(
            predictors,
            len(stimuli_subject)
        )
        
        # Fit left hemisphere
        if force_recompute or not path_lh.exists():
            print(f"  Fitting left hemisphere...")
            trf_lh = eelbrain.boosting(
                stc_lh_concatenated,
                predictors_concatenated,
                TRF_PARAMS['tmin'],
                TRF_PARAMS['tmax'],
                error=TRF_PARAMS['error'],
                basis=TRF_PARAMS['basis'],
                partitions=TRF_PARAMS['partitions'],
                test=TRF_PARAMS['test'],
                selective_stopping=TRF_PARAMS['selective_stopping']
            )
            eelbrain.save.pickle(trf_lh, path_lh)
            del trf_lh
            print(f"  Saved: {path_lh.name}")
        
        # Fit right hemisphere
        if force_recompute or not path_rh.exists():
            print(f"  Fitting right hemisphere...")
            trf_rh = eelbrain.boosting(
                stc_rh_concatenated,
                predictors_concatenated,
                TRF_PARAMS['tmin'],
                TRF_PARAMS['tmax'],
                error=TRF_PARAMS['error'],
                basis=TRF_PARAMS['basis'],
                partitions=TRF_PARAMS['partitions'],
                test=TRF_PARAMS['test'],
                selective_stopping=TRF_PARAMS['selective_stopping']
            )
            eelbrain.save.pickle(trf_rh, path_rh)
            del trf_rh
            print(f"  Saved: {path_rh.name}")
    
    print(f"\nCompleted TRF estimation for {subject}")


def _concatenate_predictors(
    predictors: List[List[Any]],
    n_stimuli: int
) -> List[Any]:
    """
    Concatenate predictors across stimuli.
    
    Args:
        predictors: List of predictor lists (one list per predictor type)
        n_stimuli: Number of stimuli
        
    Returns:
        List of concatenated predictors
    """
    predictors_concatenated = []
    
    for predictor in predictors:
        concatenated = eelbrain.concatenate([
            predictor[i] for i in range(n_stimuli)
        ])
        predictors_concatenated.append(concatenated)
    
    return predictors_concatenated


def prepare_source_data(
    stc_lh_all: List[Any],
    stc_rh_all: List[Any]
) -> tuple:
    """
    Concatenate source data across trials.
    
    Args:
        stc_lh_all: List of left hemisphere source estimates
        stc_rh_all: List of right hemisphere source estimates
        
    Returns:
        Tuple of (concatenated left hemisphere, concatenated right hemisphere)
    """
    stc_lh_concatenated = eelbrain.concatenate(stc_lh_all)
    stc_rh_concatenated = eelbrain.concatenate(stc_rh_all)
    
    return stc_lh_concatenated, stc_rh_concatenated
