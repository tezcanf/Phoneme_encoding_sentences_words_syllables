"""
Main TRF estimation pipeline.

This script orchestrates the complete TRF estimation workflow:
1. Load and preprocess data
2. Load predictors
3. Create source space representations
4. Estimate TRFs for multiple models

Usage:
    python estimate_trfs_pipeline.py --condition control_sentence --subject-index 0
    python estimate_trfs_pipeline.py --condition random_syllables --subject-index 0 --force-recompute
    python estimate_trfs_pipeline.py --condition random_word_list --subject-index 0 --region STG
"""
import argparse
from pathlib import Path

from config import (
    SUBJECTS,
    CONDITIONS,
    SOURCE_PARAMS,
    get_subject_paths,
    get_condition_paths,
    get_trf_output_dir,
)
from data_loader import (
    get_subject_stimuli,
    load_and_filter_epochs,
)
from predictors import (
    load_all_predictors,
    get_nsamples,
)
from models import get_models_for_condition
from utils import make_source_space
from trf_estimation import estimate_trfs, prepare_source_data


def run_trf_pipeline(
    condition: str,
    subject_index: int,
    force_recompute: bool = False
) -> None:
    """
    Run complete TRF estimation pipeline for one subject and condition.
    Analyzes the Superior Temporal Gyrus (STG) region.
    
    Args:
        condition: Experimental condition ('sentences', 'words', or 'syllables')
        subject_index: Index into SUBJECTS list
        force_recompute: If True, recompute existing TRFs
    """
    # Validate inputs
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}. Must be one of {CONDITIONS}")
    
    if subject_index < 0 or subject_index >= len(SUBJECTS):
        raise ValueError(f"Invalid subject index: {subject_index}. Must be 0-{len(SUBJECTS)-1}")
    
    subject = SUBJECTS[subject_index]
    print(f"\n{'='*70}")
    print(f"Starting TRF estimation pipeline")
    print(f"Subject: {subject} (index {subject_index})")
    print(f"Condition: {condition}")
    print(f"Region: STG (Superior Temporal Gyrus)")
    print(f"{'='*70}\n")
    
    # Get paths
    subject_paths = get_subject_paths(subject)
    condition_paths = get_condition_paths(condition)
    trf_output_dir = get_trf_output_dir(condition, subject)
    
    print(f"Output directory: {trf_output_dir}")
    
    # Step 1: Load stimulus order
    print("\n[1/5] Loading stimulus presentation order...")
    stimuli_subject = get_subject_stimuli(
        condition,
        subject,
        condition_paths['stimulus_dir']
    )
    print(f"  Found {len(stimuli_subject)} stimuli")
    
    # Step 2: Load epochs and handle rejected trials
    print("\n[2/5] Loading epochs and filtering rejected trials...")
    epochs, stimuli_filtered, info = load_and_filter_epochs(
        subject_paths['epochs'],
        subject_paths['events'],
        condition,
        stimuli_subject
    )
    print(f"  Retained {len(stimuli_filtered)} valid trials")
    
    # Step 3: Load predictors
    print("\n[3/5] Loading predictors...")
    
    # Determine which predictors to load based on condition
    # sentences and words conditions need word-level predictors
    include_words = condition in ['sentences', 'words']
    
    predictors = load_all_predictors(
        condition_paths['predictor_dir'],
        stimuli_filtered,
        include_phonemes=True,
        include_words=include_words
    )
    nsamples = get_nsamples(predictors)
    print(f"  Loaded {len(predictors)} predictor types")
    print(f"  Time samples: {nsamples}")
    
    # Step 4: Create source space representations
    print("\n[4/5] Creating source space representations...")
    print(f"  This may take a while...")
    
    stc_lh_all, stc_rh_all = make_source_space(
        condition=condition,
        subject=subject,
        meg_dir=subject_paths['meg_dir'],
        subjects_dir=subject_paths['subjects_dir'],
        stimuli_subject=stimuli_filtered,
        gammatone=predictors['gammatone'],
        tmax=SOURCE_PARAMS['crop_tmax'] - SOURCE_PARAMS['crop_tmin'],
        nsamples=nsamples
    )
    
    # Concatenate source data
    stc_lh_concatenated, stc_rh_concatenated = prepare_source_data(
        stc_lh_all,
        stc_rh_all
    )
    print(f"  Source data ready")
    
    # Step 5: Define models and estimate TRFs
    print("\n[5/5] Estimating TRFs...")
    
    models = get_models_for_condition(condition, predictors)
    print(f"  Estimating {len(models)} models")
    
    estimate_trfs(
        stc_lh_concatenated=stc_lh_concatenated,
        stc_rh_concatenated=stc_rh_concatenated,
        models=models,
        stimuli_subject=stimuli_filtered,
        trf_output_dir=trf_output_dir,
        subject=subject,
        force_recompute=force_recompute
    )
    
    print(f"\n{'='*70}")
    print(f"Pipeline completed successfully!")
    print(f"Output saved to: {trf_output_dir}")
    print(f"{'='*70}\n")


def main():
    """Parse command-line arguments and run pipeline."""
    parser = argparse.ArgumentParser(
        description='Estimate TRFs for MEG data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate TRFs for first subject, sentences condition
  python estimate_trfs_pipeline.py --condition sentences --subject-index 0
  
  # Estimate for syllables with forced recomputation
  python estimate_trfs_pipeline.py --condition syllables --subject-index 0 --force-recompute

All analyses use the Superior Temporal Gyrus (STG) region.
  
Available conditions: {conditions}
Available subjects: {n_subjects} subjects (indices 0-{max_index})
        """.format(
            conditions=', '.join(CONDITIONS),
            n_subjects=len(SUBJECTS),
            max_index=len(SUBJECTS)-1
        )
    )
    
    parser.add_argument(
        '--condition',
        type=str,
        required=True,
        choices=CONDITIONS,
        help='Experimental condition to analyze'
    )
    
    parser.add_argument(
        '--subject-index',
        type=int,
        required=True,
        help=f'Subject index (0-{len(SUBJECTS)-1})'
    )
    
    parser.add_argument(
        '--force-recompute',
        action='store_true',
        help='Force recomputation of existing TRFs'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    run_trf_pipeline(
        condition=args.condition,
        subject_index=args.subject_index,
        force_recompute=args.force_recompute
    )


if __name__ == '__main__':
    main()
