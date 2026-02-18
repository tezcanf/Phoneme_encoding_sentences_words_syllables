"""
Generate predictors for MEG analysis

This script generates various types of predictors:
1. Gammatone spectrograms (high-resolution)
2. Gammatone-based predictors (8-band, onset detectors)
3. Word-level predictors (cohort model features)
4. Word-level predictors with GPT features

Usage:
    python generate_predictors.py --predictor-type all
    python generate_predictors.py --predictor-type gammatone
    python generate_predictors.py --predictor-type word
"""
import os
import argparse
from pathlib import Path
import numpy as np
from eelbrain import *
from trftools.gammatone_bank import gammatone_bank
from trftools.neural import edge_detector


# ============================================================================
# Configuration
# ============================================================================

root = Path.cwd().parents[1]
DATA_ROOT = root / 'Materials'
SUBJECTS_DIR = root  / 'processed' 



# Acoustic conditions for gammatone processing
ACOUSTIC_CONDITIONS = ['random_syllables', 'control_sentence', 'random_word_list']

# Audio processing parameters
RESAMPLE_RATE = 11025  # Hz
OUTPUT_RATE = 1000     # Hz
START_TIME = 1.6       # seconds
END_TIME = 27.2        # seconds

# Gammatone parameters
GAMMATONE_FMIN = 20    # Hz
GAMMATONE_FMAX = 5000  # Hz
GAMMATONE_NBANDS = 256
GAMMATONE_NBINS = 8    # For binned predictors

# Edge detector parameter
EDGE_DETECTOR_C = 30


# ============================================================================
# Gammatone Spectrogram Generation
# ============================================================================

def generate_gammatone_spectrograms(conditions=None, overwrite=False):
    """
    Generate high-resolution gammatone spectrograms for audio stimuli.
    
    Parameters
    ----------
    conditions : list of str, optional
        List of conditions to process. If None, uses ACOUSTIC_CONDITIONS.
    overwrite : bool
        If True, regenerate existing spectrograms.
    """
    if conditions is None:
        conditions = ACOUSTIC_CONDITIONS
    
    print("=" * 70)
    print("Generating Gammatone Spectrograms")
    print("=" * 70)
    
    for condition in conditions:
        print(f"\nProcessing condition: {condition}")
        print("-" * 70)
        
        stimulus_dir = DATA_ROOT / 'Stimuli' / 'Sounds_Syllables' / condition
        
        if not stimulus_dir.exists():
            print(f"  Warning: Stimulus directory not found: {stimulus_dir}")
            continue
        
        wav_files = [f.split('.')[0] for f in os.listdir(stimulus_dir) 
                     if f.endswith('.wav')]
        
        for i, wav_name in enumerate(wav_files, 1):
            dst = stimulus_dir / f'{wav_name}-gammatone.pickle'
            
            if dst.exists() and not overwrite:
                print(f"  [{i}/{len(wav_files)}] Skipping {wav_name} (already exists)")
                continue
            
            print(f"  [{i}/{len(wav_files)}] Processing {wav_name}")
            
            # Load and preprocess audio
            wav = load.wav(stimulus_dir / f'{wav_name}.wav')
            wav = resample(wav, RESAMPLE_RATE)
            
            # Extract segment
            start_sample = int(START_TIME * RESAMPLE_RATE)
            end_sample = int(END_TIME * RESAMPLE_RATE)
            wav.x = wav.x[start_sample:end_sample]
            
            # Generate gammatone spectrogram
            gt = gammatone_bank(wav, GAMMATONE_FMIN, GAMMATONE_FMAX, 
                              GAMMATONE_NBANDS, location='left', pad=False)
            gt = resample(gt, OUTPUT_RATE)
            
            # Save
            save.pickle(gt, dst)
    
    print("\n" + "=" * 70)
    print("Gammatone spectrogram generation complete")
    print("=" * 70 + "\n")


# ============================================================================
# Gammatone-Based Predictor Generation
# ============================================================================

def generate_gammatone_predictors(conditions=None, overwrite=False):
    """
    Generate predictors based on gammatone spectrograms.
    
    Creates:
    - 8-band gammatone predictors
    - 8-band gammatone onset predictors
    
    Parameters
    ----------
    conditions : list of str, optional
        List of conditions to process. If None, uses ACOUSTIC_CONDITIONS.
    overwrite : bool
        If True, regenerate existing predictors.
    """
    if conditions is None:
        conditions = ACOUSTIC_CONDITIONS
    
    print("=" * 70)
    print("Generating Gammatone-Based Predictors")
    print("=" * 70)
    
    for condition in conditions:
        print(f"\nProcessing condition: {condition}")
        print("-" * 70)
        
        stimulus_dir = DATA_ROOT / 'Stimuli' / 'Sounds_Syllables' / condition
        predictor_dir = DATA_ROOT / 'Predictors' / condition
        
        if not stimulus_dir.exists():
            print(f"  Warning: Stimulus directory not found: {stimulus_dir}")
            continue
        
        # Create predictor directory
        predictor_dir.mkdir(exist_ok=True, parents=True)
        
        # Get list of gammatone files
        gammatone_files = [f.split('-gammatone.pickle')[0] 
                          for f in os.listdir(stimulus_dir) 
                          if f.endswith('-gammatone.pickle')]
        
        for i, wav_name in enumerate(gammatone_files, 1):
            # Check if outputs exist
            dst_gt = predictor_dir / f'{wav_name}~gammatone-8.pickle'
            dst_on = predictor_dir / f'{wav_name}~gammatone-on-8.pickle'
            
            if dst_gt.exists() and dst_on.exists() and not overwrite:
                print(f"  [{i}/{len(gammatone_files)}] Skipping {wav_name} (already exists)")
                continue
            
            print(f"  [{i}/{len(gammatone_files)}] Processing {wav_name}")
            
            # Load gammatone spectrogram
            gt = load.unpickle(stimulus_dir / f'{wav_name}-gammatone.pickle')
            
            # Remove resampling artifacts
            gt = gt.clip(0, out=gt)
            
            # Apply log transform
            gt = (gt + 1).log()
            
            # Generate onset detector
            gt_on = edge_detector(gt, c=EDGE_DETECTOR_C)
            
            # Create 8-band predictors by binning
            x_gt = gt.bin(nbins=GAMMATONE_NBINS, func=np.sum, dim='frequency')
            save.pickle(x_gt, dst_gt)
            
            x_on = gt_on.bin(nbins=GAMMATONE_NBINS, func=np.sum, dim='frequency')
            save.pickle(x_on, dst_on)
    
    print("\n" + "=" * 70)
    print("Gammatone predictor generation complete")
    print("=" * 70 + "\n")


# ============================================================================
# Word-Level Predictor Generation
# ============================================================================

def generate_word_predictors_cohort(overwrite=False):
    """
    Generate predictors for word-level variables (cohort model).
    
    Creates predictors with:
    - cohort_entropy
    - cohort_surprisal
    
    Parameters
    ----------
    overwrite : bool
        If True, regenerate existing predictors.
    """
    print("=" * 70)
    print("Generating Word-Level Predictors (Cohort Model)")
    print("=" * 70)
    
    stimulus_dir = root / 'Materials' / 'raw_data' / 'Cohort_model'
    predictor_dir = DATA_ROOT / 'Predictors'
    
    if not stimulus_dir.exists():
        print(f"Error: Stimulus directory not found: {stimulus_dir}")
        return
    
    # Create predictor directory
    predictor_dir.mkdir(exist_ok=True, parents=True)
    
    # Get list of transcription files
    csv_files = [f.split('_transcription')[0] 
                 for f in os.listdir(stimulus_dir) 
                 if f.endswith('_cohort_model.csv')]
    
    for i, segment in enumerate(csv_files, 1):
        dst = predictor_dir / f'{segment}~phoneme_cohort_model.pickle'
        
        if dst.exists() and not overwrite:
            print(f"[{i}/{len(csv_files)}] Skipping {segment} (already exists)")
            continue
        
        print(f"[{i}/{len(csv_files)}] Processing {segment}")
        
        # Load segment table
        segment_table = load.tsv(
            stimulus_dir / f'{segment}_transcription_cohort_model.csv',
            delimiter=';'
        )
        
        # Create dataset
        ds = Dataset(
            {'time': segment_table['phoneme_onset']},
            info={'tstop': segment_table[-1, 'phoneme_offset']}
        )
        
        # Add predictor variables
        for key in ['cohort_entropy', 'cohort_surprisal']:
            ds[key] = segment_table[key]
        
        # Save
        save.pickle(ds, dst)
    
    print("\n" + "=" * 70)
    print("Cohort model predictor generation complete")
    print("=" * 70 + "\n")


def generate_word_predictors_gpt(overwrite=False):
    """
    Generate predictors for word-level variables (GPT features).
    
    Creates predictors with:
    - word_number
    - word_surprisal_GPT
    - word_entropy_GPT
    
    Parameters
    ----------
    overwrite : bool
        If True, regenerate existing predictors.
    """
    print("=" * 70)
    print("Generating Word-Level Predictors (GPT Features)")
    print("=" * 70)
    
    stimulus_dir = (root / 'Materials' / 'raw_data' / 
                    'Cohort_model' / 'with_word_features')
    predictor_dir = DATA_ROOT / 'Predictors'
    
    if not stimulus_dir.exists():
        print(f"Error: Stimulus directory not found: {stimulus_dir}")
        return
    
    # Create predictor directory
    predictor_dir.mkdir(exist_ok=True, parents=True)
    
    # Get list of transcription files
    csv_files = [f.split('_transcription')[0] 
                 for f in os.listdir(stimulus_dir) 
                 if f.endswith('_cohort_model_GPT.csv')]
    
    for i, segment in enumerate(csv_files, 1):
        dst = predictor_dir / f'{segment}~phoneme_cohort_model_GPT.pickle'
        
        if dst.exists() and not overwrite:
            print(f"[{i}/{len(csv_files)}] Skipping {segment} (already exists)")
            continue
        
        print(f"[{i}/{len(csv_files)}] Processing {segment}")
        
        # Load segment table
        segment_table = load.tsv(
            stimulus_dir / f'{segment}_transcription_cohort_model_GPT.csv',
            delimiter=';'
        )
        
        # Create dataset
        ds = Dataset(
            {'time': segment_table['phoneme_onset']},
            info={'tstop': segment_table[-1, 'phoneme_offset']}
        )
        
        # Add predictor variables
        for key in ['word_number', 'word_surprisal_GPT', 'word_entropy_GPT']:
            ds[key] = segment_table[key]
        
        # Save
        save.pickle(ds, dst)
    
    print("\n" + "=" * 70)
    print("GPT predictor generation complete")
    print("=" * 70 + "\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for predictor generation."""
    parser = argparse.ArgumentParser(
        description='Generate predictors for MEG/EEG analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--predictor-type',
        choices=['all', 'gammatone', 'gammatone-spec', 'gammatone-pred', 
                 'word', 'word-cohort', 'word-gpt'],
        default='all',
        help='Type of predictors to generate'
    )
    
    parser.add_argument(
        '--conditions',
        nargs='+',
        default=None,
        help='Acoustic conditions to process (for gammatone predictors)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing predictor files'
    )
    
    args = parser.parse_args()
    
    # Generate predictors based on type
    if args.predictor_type in ['all', 'gammatone', 'gammatone-spec']:
        generate_gammatone_spectrograms(
            conditions=args.conditions,
            overwrite=args.overwrite
        )
    
    if args.predictor_type in ['all', 'gammatone', 'gammatone-pred']:
        generate_gammatone_predictors(
            conditions=args.conditions,
            overwrite=args.overwrite
        )
    
    if args.predictor_type in ['all', 'word', 'word-cohort']:
        generate_word_predictors_cohort(overwrite=args.overwrite)
    
    if args.predictor_type in ['all', 'word', 'word-gpt']:
        generate_word_predictors_gpt(overwrite=args.overwrite)
    
    print("\n" + "=" * 70)
    print("All requested predictors generated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()