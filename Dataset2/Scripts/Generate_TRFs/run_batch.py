#!/usr/bin/env python3
"""
Batch Configuration Script

This script provides an easy way to run TRF estimation without typing long
command-line arguments. Just edit the settings below and run:

    python run_batch.py

This is particularly useful for:
- Running the same configuration repeatedly
- Documenting your analysis workflow
- Avoiding typos in command-line arguments
"""

import sys
import subprocess

# ============================================================================
# CONFIGURATION - EDIT THESE SETTINGS
# ============================================================================

# What to process:
# Options: 'single', 'multiple', 'dutch', 'chinese', 'all'
MODE = 'single'  # <-- CHANGE THIS

# If MODE = 'single' or 'multiple', specify subjects here:
SUBJECTS = ['sub-003']  # <-- CHANGE THIS

# Which conditions to process:
# Options: ['Words'], ['Syllables'], ['Words', 'Syllables']
CONDITIONS = ['Words']  # <-- CHANGE THIS

# Stimulus type:
# Options: 'native', 'non-native', 'Dutch', 'Chinese'
#   - 'native': Dutch subjects → Dutch stimuli, Chinese subjects → Chinese stimuli
#   - 'non-native': Dutch subjects → Chinese stimuli, Chinese subjects → Dutch stimuli
#   - 'Dutch': All subjects → Dutch stimuli
#   - 'Chinese': All subjects → Chinese stimuli
STIMULUS_TYPE = 'native'  # <-- CHANGE THIS

# ============================================================================
# DON'T EDIT BELOW THIS LINE (unless you know what you're doing)
# ============================================================================

def build_command():
    """Build the command-line arguments for estimate_trfs_unified.py"""
    cmd = ['python', 'estimate_trfs_unified.py']
    
    # Add subject/group arguments
    if MODE == 'single' or MODE == 'multiple':
        if not SUBJECTS:
            print("ERROR: SUBJECTS list is empty but MODE is 'single' or 'multiple'")
            print("Please specify at least one subject in the SUBJECTS list")
            sys.exit(1)
        cmd.extend(['--subjects'] + SUBJECTS)
        
    elif MODE == 'dutch':
        cmd.extend(['--group', 'dutch'])
        
    elif MODE == 'chinese':
        cmd.extend(['--group', 'chinese'])
        
    elif MODE == 'all':
        cmd.append('--all')
        
    else:
        print(f"ERROR: Unknown MODE '{MODE}'")
        print("Valid options: 'single', 'multiple', 'dutch', 'chinese', 'all'")
        sys.exit(1)
    
    # Add conditions
    if CONDITIONS:
        cmd.extend(['--conditions'] + CONDITIONS)
    else:
        print("WARNING: No conditions specified, will use default (all conditions)")
    
    # Add stimulus type
    cmd.extend(['--stimulus-type', STIMULUS_TYPE])
    
    return cmd


def validate_configuration():
    """Validate the configuration before running"""
    issues = []
    
    # Check MODE
    if MODE not in ['single', 'multiple', 'dutch', 'chinese', 'all']:
        issues.append(f"Invalid MODE: '{MODE}'")
    
    # Check SUBJECTS for single/multiple mode
    if MODE in ['single', 'multiple'] and not SUBJECTS:
        issues.append("SUBJECTS list is empty but MODE requires subjects")
    
    # Check CONDITIONS
    valid_conditions = ['Words', 'Syllables']
    for cond in CONDITIONS:
        if cond not in valid_conditions:
            issues.append(f"Invalid condition: '{cond}' (must be 'Words' or 'Syllables')")
    
    # Check STIMULUS_TYPE
    valid_stim_types = ['native', 'non-native', 'Dutch', 'Chinese']
    if STIMULUS_TYPE not in valid_stim_types:
        issues.append(f"Invalid STIMULUS_TYPE: '{STIMULUS_TYPE}'")
    
    return issues


def print_configuration():
    """Print the current configuration"""
    print("="*80)
    print("BATCH TRF ESTIMATION RUNNER")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Mode: {MODE}")
    
    if MODE in ['single', 'multiple']:
        print(f"  Subjects: {SUBJECTS}")
    elif MODE == 'dutch':
        print(f"  Subjects: All Dutch subjects")
    elif MODE == 'chinese':
        print(f"  Subjects: All Chinese subjects")
    elif MODE == 'all':
        print(f"  Subjects: All subjects (Dutch + Chinese)")
    
    print(f"  Conditions: {CONDITIONS}")
    print(f"  Stimulus type: {STIMULUS_TYPE}")


def main():
    # Print configuration
    print_configuration()
    
    # Validate configuration
    issues = validate_configuration()
    if issues:
        print("\n" + "="*80)
        print("CONFIGURATION ERRORS:")
        print("="*80)
        for issue in issues:
            print(f"  ✗ {issue}")
        print("\nPlease fix the issues above and try again.")
        sys.exit(1)
    
    # Build command
    cmd = build_command()
    
    print("\n" + "="*80)
    print("Running command:")
    print(" ".join(cmd))
    print("="*80 + "\n")
    
    # Run the command
    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running command: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
