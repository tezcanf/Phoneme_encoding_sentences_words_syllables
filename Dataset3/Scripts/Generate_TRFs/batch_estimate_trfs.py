"""
Batch processing script for TRF estimation.

Run TRF estimation for multiple subjects and/or conditions.

Usage:
    # Run all subjects for one condition
    python batch_estimate_trfs.py --condition control_sentence --all-subjects
    
    # Run specific subject range
    python batch_estimate_trfs.py --condition control_sentence --subject-range 0 5
    
    # Run all conditions for one subject
    python batch_estimate_trfs.py --subject-index 0 --all-conditions
    
    # Run multiple specific subjects
    python batch_estimate_trfs.py --condition control_sentence --subject-indices 0 1 2 3
"""
import argparse
import sys
from pathlib import Path

from config import SUBJECTS, CONDITIONS
from estimate_trfs_pipeline import run_trf_pipeline


def batch_process(
    conditions: list = None,
    subject_indices: list = None,
    force_recompute: bool = False
) -> None:
    """
    Run TRF estimation for multiple subjects and/or conditions.
    All analyses use the Superior Temporal Gyrus (STG) region.
    
    Args:
        conditions: List of condition names ('sentences', 'words', 'syllables')
        subject_indices: List of subject indices (None = all subjects)
        force_recompute: If True, recompute existing TRFs
    """
    # Use all conditions if not specified
    if conditions is None:
        conditions = CONDITIONS
    
    # Use all subjects if not specified
    if subject_indices is None:
        subject_indices = list(range(len(SUBJECTS)))
    
    # Calculate total number of jobs
    total_jobs = len(conditions) * len(subject_indices)
    current_job = 0
    failed_jobs = []
    
    print(f"\n{'='*70}")
    print(f"BATCH TRF ESTIMATION")
    print(f"{'='*70}")
    print(f"Conditions: {len(conditions)} ({', '.join(conditions)})")
    print(f"Subjects: {len(subject_indices)} (indices: {subject_indices})")
    print(f"Region: STG (Superior Temporal Gyrus)")
    print(f"Total jobs: {total_jobs}")
    print(f"{'='*70}\n")
    
    # Process each combination
    for condition in conditions:
        for subject_idx in subject_indices:
            current_job += 1
            subject = SUBJECTS[subject_idx]
            
            print(f"\n{'#'*70}")
            print(f"JOB {current_job}/{total_jobs}")
            print(f"Subject: {subject} (index {subject_idx})")
            print(f"Condition: {condition}")
            print(f"{'#'*70}")
            
            try:
                run_trf_pipeline(
                    condition=condition,
                    subject_index=subject_idx,
                    force_recompute=force_recompute
                )
            except Exception as e:
                print(f"\n{'!'*70}")
                print(f"ERROR in job {current_job}/{total_jobs}")
                print(f"Subject: {subject}, Condition: {condition}")
                print(f"Error: {str(e)}")
                print(f"{'!'*70}\n")
                failed_jobs.append((subject, condition, str(e)))
                continue
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total jobs: {total_jobs}")
    print(f"Successful: {total_jobs - len(failed_jobs)}")
    print(f"Failed: {len(failed_jobs)}")
    
    if failed_jobs:
        print(f"\nFailed jobs:")
        for subject, condition, error in failed_jobs:
            print(f"  - {subject}, {condition}: {error}")
    
    print(f"{'='*70}\n")


def main():
    """Parse command-line arguments and run batch processing."""
    parser = argparse.ArgumentParser(
        description='Batch TRF estimation for multiple subjects/conditions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all subjects for sentences condition
  python batch_estimate_trfs.py --condition sentences --all-subjects
  
  # Run first 5 subjects
  python batch_estimate_trfs.py --condition sentences --subject-range 0 5
  
  # Run all conditions for first subject
  python batch_estimate_trfs.py --subject-index 0 --all-conditions
  
  # Run specific subjects
  python batch_estimate_trfs.py --condition sentences --subject-indices 0 1 2 3 4
  
  # Run multiple conditions
  python batch_estimate_trfs.py --conditions sentences syllables --all-subjects

All analyses use the Superior Temporal Gyrus (STG) region.
        """
    )
    
    # Condition selection
    condition_group = parser.add_mutually_exclusive_group()
    condition_group.add_argument(
        '--condition',
        type=str,
        choices=CONDITIONS,
        help='Single condition to analyze'
    )
    condition_group.add_argument(
        '--conditions',
        type=str,
        nargs='+',
        choices=CONDITIONS,
        help='Multiple conditions to analyze'
    )
    condition_group.add_argument(
        '--all-conditions',
        action='store_true',
        help='Analyze all conditions'
    )
    
    # Subject selection
    subject_group = parser.add_mutually_exclusive_group()
    subject_group.add_argument(
        '--subject-index',
        type=int,
        help='Single subject index'
    )
    subject_group.add_argument(
        '--subject-indices',
        type=int,
        nargs='+',
        help='Multiple subject indices'
    )
    subject_group.add_argument(
        '--subject-range',
        type=int,
        nargs=2,
        metavar=('START', 'END'),
        help='Range of subject indices (inclusive start, exclusive end)'
    )
    subject_group.add_argument(
        '--all-subjects',
        action='store_true',
        help='Analyze all subjects'
    )
    
    # Other options
    parser.add_argument(
        '--force-recompute',
        action='store_true',
        help='Force recomputation of existing TRFs'
    )
    
    args = parser.parse_args()
    
    # Determine conditions to process
    if args.all_conditions:
        conditions = CONDITIONS
    elif args.conditions:
        conditions = args.conditions
    elif args.condition:
        conditions = [args.condition]
    else:
        print("Error: Must specify condition(s) to process")
        parser.print_help()
        sys.exit(1)
    
    # Determine subjects to process
    if args.all_subjects:
        subject_indices = list(range(len(SUBJECTS)))
    elif args.subject_range:
        start, end = args.subject_range
        subject_indices = list(range(start, end))
    elif args.subject_indices:
        subject_indices = args.subject_indices
    elif args.subject_index is not None:
        subject_indices = [args.subject_index]
    else:
        print("Error: Must specify subject(s) to process")
        parser.print_help()
        sys.exit(1)
    
    # Validate subject indices
    invalid_indices = [i for i in subject_indices if i < 0 or i >= len(SUBJECTS)]
    if invalid_indices:
        print(f"Error: Invalid subject indices: {invalid_indices}")
        print(f"Valid range: 0-{len(SUBJECTS)-1}")
        sys.exit(1)
    
    # Run batch processing
    batch_process(
        conditions=conditions,
        subject_indices=subject_indices,
        force_recompute=args.force_recompute
    )


if __name__ == '__main__':
    main()
