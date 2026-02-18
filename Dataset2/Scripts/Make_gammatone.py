"""
Generate gammatone spectrograms and derived predictors

This script performs two steps:
1. Generate high-resolution gammatone spectrograms from audio files
2. Process spectrograms into frequency-binned predictors with onset detection
"""
from pathlib import Path
import os
import numpy as np
from eelbrain import *
from trftools.gammatone_bank import gammatone_bank
from trftools.neural import edge_detector


# Configuration

# Paths
root = Path.cwd().parents[0]
DATA_ROOT = root / 'Materials'


# Dutch_stimuli or Chinese_stimuli; OR Dutch_participants or Chinese_participants
# Words or Syllables

STIMULUS_DIR = DATA_ROOT / 'Dutch_stimuli' / 'Stimuli' / 'Dutch_participants' / 'Words' 
PREDICTOR_DIR = DATA_ROOT /  'Dutch_stimuli' / 'Predictors' / 'Dutch_participants' / 'Words'

# Gammatone parameters
GT_FREQ_MIN = 20
GT_FREQ_MAX = 5000
GT_N_CHANNELS = 256
GT_RESAMPLE_RATE = 1000

# Predictor parameters
GT_N_BINS = 8
ONSET_DETECTOR_C = 30

# Ensure output directory exists
PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)

# Get list of audio files
list_wav = [f.split('.')[0] for f in os.listdir(STIMULUS_DIR) if f.endswith('.wav')]
print(dur)

def generate_gammatone_spectrogram(wav_name):
    """Generate gammatone spectrogram for a single audio file"""
    dst = PREDICTOR_DIR / f'{wav_name}-gammatone.pickle'
    if dst.exists():
        return True
    
    print(f"Generating gammatone for: {wav_name}")
    wav = load.wav(STIMULUS_DIR / f'{wav_name}.wav')
    wav = resample(wav, 11025)
    gt = gammatone_bank(wav, GT_FREQ_MIN, GT_FREQ_MAX, GT_N_CHANNELS, 
                        location='left', pad=False)
    gt = resample(gt, GT_RESAMPLE_RATE)
    save.pickle(gt, dst)
    return False


def generate_gammatone_predictors(wav_name):
    """Generate frequency-binned predictors from gammatone spectrogram"""
    dst_main = PREDICTOR_DIR / f'{wav_name}~gammatone-{GT_N_BINS}.pickle'
    dst_onset = PREDICTOR_DIR / f'{wav_name}~gammatone-on-{GT_N_BINS}.pickle'
    
    if dst_main.exists() and dst_onset.exists():
        return True
    
    print(f"Generating predictors for: {wav_name}")
    
    # Load gammatone spectrogram
    gt = load.unpickle(PREDICTOR_DIR / f'{wav_name}-gammatone.pickle')
    
    # Remove resampling artifacts
    gt = gt.clip(0, out=gt)
    
    # Apply log transform
    gt = (gt + 1).log()
    
    # Generate onset detector model
    gt_on = edge_detector(gt, c=ONSET_DETECTOR_C)
    
    # Create frequency-binned predictors
    x_main = gt.bin(nbins=GT_N_BINS, func=np.sum, dim='frequency')
    save.pickle(x_main, dst_main)
    
    x_onset = gt_on.bin(nbins=GT_N_BINS, func=np.sum, dim='frequency')
    save.pickle(x_onset, dst_onset)
    
    return False


def main():
    """Run the complete gammatone processing pipeline"""
    print(f"Processing {len(list_wav)} audio files...")
    print(f"Output directory: {PREDICTOR_DIR}")
    print()
    
    # Step 1: Generate gammatone spectrograms
    print("=" * 60)
    print("STEP 1: Generating gammatone spectrograms")
    print("=" * 60)
    gt_skipped = 0
    for wav_name in list_wav:
        if generate_gammatone_spectrogram(wav_name):
            gt_skipped += 1
    print(f"Completed: {len(list_wav) - gt_skipped} generated, {gt_skipped} skipped (already exist)")
    print()
    
    # Step 2: Generate predictors
    print("=" * 60)
    print("STEP 2: Generating frequency-binned predictors")
    print("=" * 60)
    pred_skipped = 0
    for wav_name in list_wav:
        if generate_gammatone_predictors(wav_name):
            pred_skipped += 1
    print(f"Completed: {len(list_wav) - pred_skipped} generated, {pred_skipped} skipped (already exist)")
    print()
    
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()