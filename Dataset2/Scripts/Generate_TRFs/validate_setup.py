#!/usr/bin/env python3
"""
Test and Validation Script

This script validates the configuration and checks that all required files exist
before running TRF estimation.
"""

import os
import sys
from pathlib import Path
from config import (
    DUTCH_SUBJECTS, CHINESE_SUBJECTS, CONDITIONS,
    get_subject_config
)


def check_directory_exists(path, description):
    """Check if a directory exists and is readable"""
    if not path.exists():
        print(f"  ✗ {description}: {path} (NOT FOUND)")
        return False
    elif not os.access(path, os.R_OK):
        print(f"  ✗ {description}: {path} (NOT READABLE)")
        return False
    else:
        print(f"  ✓ {description}: {path}")
        return True


def check_file_exists(path, description):
    """Check if a file exists and is readable"""
    if not path.exists():
        print(f"  ✗ {description}: {path} (NOT FOUND)")
        return False
    elif not os.access(path, os.R_OK):
        print(f"  ✗ {description}: {path} (NOT READABLE)")
        return False
    else:
        print(f"  ✓ {description}")
        return True


def check_stimulus_files(stimulus_dir, subject):
    """Check if stimulus files exist for a subject"""
    if not stimulus_dir.exists():
        return False, 0
    
    files = [f for f in os.listdir(stimulus_dir) 
             if (subject in f) and (f.endswith('.wav'))]
    
    return len(files) > 0, len(files)


def check_predictor_files(predictor_dir, subject):
    """Check if predictor files exist for a subject"""
    if not predictor_dir.exists():
        return False, []
    
    # List of required predictor types
    required_predictors = [
        'gammatone-8.pickle',
        'gammatone-on-8.pickle',
        'phoneme_cohort_model.pickle'
    ]
    
    missing = []
    for pred_type in required_predictors:
        matching_files = [f for f in os.listdir(predictor_dir) 
                         if pred_type in f and subject in f]
        if len(matching_files) == 0:
            missing.append(pred_type)
    
    return len(missing) == 0, missing


def check_meg_files(meg_dir, subject, stimulus_type):
    """Check if MEG files exist for a subject"""
    if not meg_dir.exists():
        return False, []
    
    # MEG filename depends on stimulus type
    meg_filename = f'{subject}_resampled_300Hz-1-100Hz_filtered-ICA-eyeblink_{stimulus_type}_stimuli.fif'
    
    required_files = [
        meg_filename,
        f'{subject}_forward.fif',
        f'{subject}_inverse.fif'
    ]
    
    missing = []
    for filename in required_files:
        if not (meg_dir / filename).exists():
            missing.append(filename)
    
    return len(missing) == 0, missing


def check_events_file(subjects_dir, subject, stimulus_type):
    """Check if events file exists for a subject"""
    events_file = subjects_dir / subject / 'meg' / f'{subject}-eve_{stimulus_type}_stimuli.fif'
    
    if events_file.exists():
        return True, None
    else:
        return False, str(events_file)


def validate_subject_condition(subject, condition, config):
    """
    Validate all required files for a subject-condition combination
    
    Parameters
    ----------
    subject : str
        Subject ID
    condition : str
        Condition name ('Words' or 'Syllables')
    config : dict
        Configuration dictionary from get_subject_config
    
    Returns
    -------
    bool
        True if all files exist, False otherwise
    """
    print(f"\n  Checking {subject} - {condition} ({config['stimulus_type']} stimuli):")
    
    all_ok = True
    
    # Check stimulus directory
    stimulus_dir = config['stimuli_dir'] / condition
    if not check_directory_exists(stimulus_dir, "Stimulus directory"):
        all_ok = False
    else:
        has_files, n_files = check_stimulus_files(stimulus_dir, subject)
        if has_files:
            print(f"    ✓ Found {n_files} stimulus files")
        else:
            print(f"    ✗ No stimulus files found for {subject}")
            all_ok = False
    
    # Check predictor directory
    predictor_dir = config['predictors_dir'] / condition
    if not check_directory_exists(predictor_dir, "Predictor directory"):
        all_ok = False
    else:
        has_files, missing = check_predictor_files(predictor_dir, subject)
        if has_files:
            print(f"    ✓ All predictor files found")
        else:
            print(f"    ✗ Missing predictor files: {', '.join(missing)}")
            all_ok = False
    
    # Check MEG directory
    meg_dir = config['meg_root'] / subject / 'meg'
    if not check_directory_exists(meg_dir, "MEG directory"):
        all_ok = False
    else:
        has_files, missing = check_meg_files(meg_dir, subject, config['stimulus_type'])
        if has_files:
            print(f"    ✓ All MEG files found")
        else:
            print(f"    ✗ Missing MEG files: {', '.join(missing)}")
            all_ok = False
    
    # Check subjects_dir (FreeSurfer anatomy)
    subject_dir = config['subjects_dir'] / subject
    if not check_directory_exists(subject_dir, "Subject anatomy directory"):
        all_ok = False
    
    # Check events file
    has_events, events_path = check_events_file(config['subjects_dir'], subject, 
                                                 config['stimulus_type'])
    if has_events:
        print(f"    ✓ Events file found")
    else:
        print(f"    ✗ Missing events file: {events_path}")
        all_ok = False
    
    return all_ok


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate TRF estimation configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--subjects', nargs='+', 
                       help='Specific subjects to validate')
    parser.add_argument('--group', choices=['dutch', 'chinese', 'all'], 
                       help='Validate all subjects in a group')
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS,
                       default=CONDITIONS, 
                       help='Conditions to validate (default: all)')
    parser.add_argument('--stimulus-type', 
                       choices=['native', 'non-native', 'Dutch', 'Chinese', 'all'],
                       default='native',
                       help='Stimulus type to validate (default: native). Use "all" to check all combinations.')
    parser.add_argument('--quick', action='store_true',
                       help='Quick check (only first subject per group)')
    parser.add_argument('--check-cross-linguistic', action='store_true',
                       help='Check both native and non-native stimulus combinations')
    
    args = parser.parse_args()
    
    # Determine which subjects to validate
    if args.group == 'all' or (not args.subjects and not args.group):
        subjects = DUTCH_SUBJECTS + CHINESE_SUBJECTS
    elif args.group == 'dutch':
        subjects = DUTCH_SUBJECTS
    elif args.group == 'chinese':
        subjects = CHINESE_SUBJECTS
    elif args.subjects:
        subjects = args.subjects
    else:
        subjects = []
    
    if args.quick:
        if args.group == 'dutch' or args.group == 'all':
            subjects = [DUTCH_SUBJECTS[0]] + ([CHINESE_SUBJECTS[0]] if args.group == 'all' else [])
        elif args.group == 'chinese':
            subjects = [CHINESE_SUBJECTS[0]]
    
    conditions = args.conditions
    
    # Determine which stimulus types to validate
    if args.check_cross_linguistic:
        stimulus_types = ['native', 'non-native']
        print(f"Checking both native and non-native stimulus combinations")
    elif args.stimulus_type == 'all':
        stimulus_types = ['Dutch', 'Chinese']
        print(f"Checking all stimulus types (Dutch and Chinese)")
    else:
        stimulus_types = [args.stimulus_type]
    
    # Print header
    print("="*80)
    print("TRF ESTIMATION VALIDATION")
    print("="*80)
    print(f"\nValidating {len(subjects)} subject(s) × {len(conditions)} condition(s) × {len(stimulus_types)} stimulus type(s)")
    print(f"Stimulus types: {', '.join(stimulus_types)}")
    
    # Track results
    all_valid = True
    results = {}
    
    # Validate each combination
    for stimulus_type in stimulus_types:
        print(f"\n{'='*80}")
        print(f"STIMULUS TYPE: {stimulus_type.upper()}")
        print(f"{'='*80}")
        
        for subject in subjects:
            print(f"\n{'─'*80}")
            print(f"Subject: {subject}")
            subject_group = "Dutch" if subject in DUTCH_SUBJECTS else "Chinese"
            print(f"Group: {subject_group}")
            print(f"{'─'*80}")
            
            subject_valid = True
            
            for condition in conditions:
                # Get configuration based on stimulus type
                config = get_subject_config(subject, stimulus_type)
                
                # Validate this subject-condition combination
                is_valid = validate_subject_condition(subject, condition, config)
                subject_valid = subject_valid and is_valid
                
                # Store result
                key = f"{subject}_{condition}_{config['stimulus_type']}"
                results[key] = is_valid
            
            all_valid = all_valid and subject_valid
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nTotal combinations tested: {total}")
    print(f"Passed: {passed}/{total}")
    
    # Breakdown by stimulus type
    if len(stimulus_types) > 1:
        print(f"\nBreakdown by stimulus type:")
        for stim_type in stimulus_types:
            stim_results = {k: v for k, v in results.items() if stim_type in k or 
                          (stim_type == 'native' and ('Dutch_stimuli' in k or 'Chinese_stimuli' in k)) or
                          (stim_type == 'non-native' and ('Dutch_stimuli' in k or 'Chinese_stimuli' in k))}
            stim_passed = sum(1 for v in stim_results.values() if v)
            stim_total = len(stim_results)
            print(f"  {stim_type}: {stim_passed}/{stim_total} passed")
    
    if not all_valid:
        print("\n✗ Validation FAILED - see errors above")
        print("\nFailed combinations:")
        for key, value in results.items():
            if not value:
                print(f"  - {key}")
        print("\nPlease check that:")
        print("  1. All data files are in the correct directories")
        print("  2. File naming conventions are followed")
        print("  3. Paths in config.py point to the correct locations")
        sys.exit(1)
    else:
        print("\n✓ Validation PASSED - all required files found")
        print("\nYou can now run TRF estimation with:")
        if len(stimulus_types) == 1:
            print(f"  python estimate_trfs_unified.py --group all --stimulus-type {args.stimulus_type}")
        else:
            print(f"  python estimate_trfs_unified.py --group all --stimulus-type native")
            print(f"  python estimate_trfs_unified.py --group all --stimulus-type non-native")
        sys.exit(0)


if __name__ == '__main__':
    main()