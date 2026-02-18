"""
Generate predictors for TRF analysis

This script generates three types of predictors:
1. Gammatone spectrograms (acoustic features)
2. Gammatone-derived predictors (1-band and 8-band, with onset detection)
3. Word-level predictors (cohort model and GPT-2 features)

Run sections independently as needed.
"""

from pathlib import Path
import os
import numpy as np
from eelbrain import *
from trftools import gammatone_bank
from trftools.neural import edge_detector


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Central configuration for all predictor generation"""
    
    # Root directories
    root = Path.cwd().parents[0]
    DATA_ROOT = root / 'Materials' 
    ALT_DATA_ROOT = root 
    
    # Stimulus directories
    WORD_LIST_STIMULI = DATA_ROOT / "Stimuli" / "Word_list"
    SENTENCE_STIMULI = DATA_ROOT / "Stimuli" / "Sentences"
    
    # Predictor output directories
    PREDICTOR_ROOT = DATA_ROOT / "Predictors"
    WORD_PREDICTOR_DIR = PREDICTOR_ROOT / "Word_list"
    SENTENCE_PREDICTOR_DIR = PREDICTOR_ROOT / "Sentences"
    
    # Raw data directories for word-level features
    # Word_list
    COHORT_MODEL_DIR = DATA_ROOT / 'Cohort_model' / 'Word_list' 
    GPT2_MODEL_DIR = DATA_ROOT / 'GPT2' / 'Word_list' 
    
    # Sentences
    COHORT_MODEL_SENTENCES_DIR = DATA_ROOT / 'Cohort_model' / 'Sentences'  
    GPT2_SENTENCES_DIR = DATA_ROOT / 'GPT2' / 'Sentences' 
    
    # Gammatone parameters
    GAMMATONE_FMIN = 20
    GAMMATONE_FMAX = 5000
    GAMMATONE_N_BANDS = 256
    GAMMATONE_RESAMPLE_RATE = 11025
    GAMMATONE_OUTPUT_RATE = 1000
    
    # Edge detector parameter
    EDGE_DETECTOR_C = 30
    
    # Frequency binning
    N_BINS = 8


# =============================================================================
# Utility functions
# =============================================================================

def get_wav_files(directory):
    """Get list of wav file basenames from directory"""
    if not directory.exists():
        print(f"Warning: Directory does not exist: {directory}")
        return []
    return [f.split('.')[0] for f in os.listdir(directory) if f.endswith('.wav')]


def get_cohort_model_files(directory):
    """Get list of cohort model CSV basenames"""
    if not directory.exists():
        print(f"Warning: Directory does not exist: {directory}")
        return []
    suffix = '_words_phonemes_revised_revised_cohort_model.csv'
    return [f.replace(suffix, '') for f in os.listdir(directory) if f.endswith(suffix)]


def get_gpt_model_files(directory):
    """Get list of GPT model CSV basenames"""
    if not directory.exists():
        print(f"Warning: Directory does not exist: {directory}")
        return []
    suffix = '_words_phonemes_revised_revised_cohort_model_GPT.csv'
    return [f.replace(suffix, '') for f in os.listdir(directory) if f.endswith(suffix)]


# =============================================================================
# Gammatone spectrogram generation
# =============================================================================

def generate_gammatone_spectrograms(stimulus_dir, file_list=None):
    """
    Generate high-resolution gammatone spectrograms from WAV files
    
    Parameters
    ----------
    stimulus_dir : Path
        Directory containing WAV files
    file_list : list, optional
        List of file basenames to process. If None, process all WAV files.
    """
    print(f"\n{'='*70}")
    print("GENERATING GAMMATONE SPECTROGRAMS")
    print(f"{'='*70}")
    print(f"Stimulus directory: {stimulus_dir}")
    
    if file_list is None:
        file_list = get_wav_files(stimulus_dir)
    
    if not file_list:
        print("No files to process")
        return
    
    print(f"Files to process: {len(file_list)}")
    
    processed = 0
    skipped = 0
    
    for filename in file_list:
        dst = stimulus_dir / f'{filename}-gammatone.pickle'
        
        if dst.exists():
            skipped += 1
            continue
        
        print(f"Processing: {filename}")
        
        # Load and resample audio
        wav = load.wav(stimulus_dir / f'{filename}.wav')
        wav = resample(wav, Config.GAMMATONE_RESAMPLE_RATE)
        
        # Generate gammatone spectrogram
        gt = gammatone_bank(
            wav, 
            Config.GAMMATONE_FMIN, 
            Config.GAMMATONE_FMAX, 
            Config.GAMMATONE_N_BANDS, 
            location='left', 
            pad=False
        )
        
        # Resample to output rate
        gt = resample(gt, Config.GAMMATONE_OUTPUT_RATE)
        
        # Save
        save.pickle(gt, dst)
        processed += 1
    
    print(f"\nProcessed: {processed} files")
    print(f"Skipped (already exist): {skipped} files")


# =============================================================================
# Gammatone-based predictor generation
# =============================================================================

def generate_gammatone_predictors(stimulus_dir, predictor_dir, file_list=None):
    """
    Generate predictors based on gammatone spectrograms
    
    Creates four types of predictors:
    - gammatone-1: 1-band summed gammatone
    - gammatone-on-1: 1-band onset detector
    - gammatone-8: 8-band binned gammatone
    - gammatone-on-8: 8-band onset detector
    
    Parameters
    ----------
    stimulus_dir : Path
        Directory containing gammatone pickle files
    predictor_dir : Path
        Output directory for predictors
    file_list : list, optional
        List of file basenames to process. If None, process all WAV files.
    """
    print(f"\n{'='*70}")
    print("GENERATING GAMMATONE-BASED PREDICTORS")
    print(f"{'='*70}")
    print(f"Stimulus directory: {stimulus_dir}")
    print(f"Predictor directory: {predictor_dir}")
    
    # Ensure output directory exists
    predictor_dir.mkdir(parents=True, exist_ok=True)
    
    if file_list is None:
        file_list = get_wav_files(stimulus_dir)
    
    if not file_list:
        print("No files to process")
        return
    
    print(f"Files to process: {len(file_list)}")
    
    for filename in file_list:
        gt_file = stimulus_dir / f'{filename}-gammatone.pickle'
        
        if not gt_file.exists():
            print(f"Warning: Gammatone file not found for {filename}, skipping")
            continue
        
        print(f"Processing: {filename}")
        
        # Load gammatone spectrogram
        gt = load.unpickle(gt_file)
        
        # Preprocessing: remove resampling artifacts and apply log transform
        gt = gt.clip(0, out=gt)
        gt = (gt + 1).log()
        
        # Generate onset detector
        gt_on = edge_detector(gt, c=Config.EDGE_DETECTOR_C)
        
        # 1-band predictors (sum across all frequencies)
        save.pickle(
            gt.sum('frequency'), 
            predictor_dir / f'{filename}~gammatone-1.pickle'
        )
        save.pickle(
            gt_on.sum('frequency'), 
            predictor_dir / f'{filename}~gammatone-on-1.pickle'
        )
        
        # 8-band predictors (bin frequencies into 8 bands)
        gt_8 = gt.bin(nbins=Config.N_BINS, func=np.sum, dim='frequency')
        save.pickle(gt_8, predictor_dir / f'{filename}~gammatone-8.pickle')
        
        gt_on_8 = gt_on.bin(nbins=Config.N_BINS, func=np.sum, dim='frequency')
        save.pickle(gt_on_8, predictor_dir / f'{filename}~gammatone-on-8.pickle')
    
    print(f"\nCompleted processing {len(file_list)} files")


def generate_all_gammatone_predictors():
    """Generate gammatone predictors for both Word_list and Sentences conditions"""
    config = Config()
    
    print(f"\n{'='*70}")
    print("GENERATING GAMMATONE PREDICTORS FOR ALL CONDITIONS")
    print(f"{'='*70}")
    
    # Word_list condition
    print("\n--- Processing Word_list condition ---")
    generate_gammatone_predictors(
        stimulus_dir=config.WORD_LIST_STIMULI,
        predictor_dir=config.WORD_PREDICTOR_DIR
    )
    
    # Sentences condition
    print("\n--- Processing Sentences condition ---")
    generate_gammatone_predictors(
        stimulus_dir=config.SENTENCE_STIMULI,
        predictor_dir=config.SENTENCE_PREDICTOR_DIR
    )


# =============================================================================
# Word-level predictor generation
# =============================================================================

def generate_cohort_model_predictors(data_dir, predictor_dir, file_list=None):
    """
    Generate predictors for cohort model features
    
    Creates predictors with:
    - cohort_entropy
    - cohort_surprisal
    
    Parameters
    ----------
    data_dir : Path
        Directory containing cohort model CSV files
    predictor_dir : Path
        Output directory for predictors
    file_list : list, optional
        List of file basenames to process. If None, process all CSV files.
    """
    print(f"\n{'='*70}")
    print("GENERATING COHORT MODEL PREDICTORS")
    print(f"{'='*70}")
    print(f"Data directory: {data_dir}")
    print(f"Predictor directory: {predictor_dir}")
    
    # Ensure output directory exists
    predictor_dir.mkdir(parents=True, exist_ok=True)
    
    if file_list is None:
        file_list = get_cohort_model_files(data_dir)
    
    if not file_list:
        print("No files to process")
        return
    
    print(f"Files to process: {len(file_list)}")
    
    for segment in file_list:
        csv_file = data_dir / f'{segment}_words_phonemes_revised_revised_cohort_model.csv'
        
        if not csv_file.exists():
            print(f"Warning: CSV file not found for {segment}, skipping")
            continue
        
        print(f"Processing: {segment}")
        
        # Load segment table
        segment_table = load.tsv(csv_file, delimiter=';')
        
        # Create dataset with timing information
        ds = Dataset(
            {'time': segment_table['phoneme_onset']}, 
            info={'tstop': segment_table[-1, 'phoneme_offset']}
        )
        
        # Add predictor variables
        for key in ['cohort_entropy', 'cohort_surprisal']:
            ds[key] = segment_table[key]
        
        # Save
        output_file = predictor_dir / f'{segment}~phoneme_cohort_model.pickle'
        save.pickle(ds, output_file)
    
    print(f"\nCompleted processing {len(file_list)} files")


def generate_all_cohort_model_predictors():
    """Generate cohort model predictors for both Word_list and Sentences conditions"""
    config = Config()
    
    print(f"\n{'='*70}")
    print("GENERATING COHORT MODEL PREDICTORS FOR ALL CONDITIONS")
    print(f"{'='*70}")
    
    # Word_list condition
    print("\n--- Processing Word_list condition ---")
    generate_cohort_model_predictors(
        data_dir=config.COHORT_MODEL_DIR,
        predictor_dir=config.WORD_PREDICTOR_DIR
    )
    
    # Sentences condition
    print("\n--- Processing Sentences condition ---")
    if config.COHORT_MODEL_SENTENCES_DIR.exists():
        generate_cohort_model_predictors(
            data_dir=config.COHORT_MODEL_SENTENCES_DIR,
            predictor_dir=config.SENTENCE_PREDICTOR_DIR
        )
    else:
        print(f"Warning: Sentences directory not found: {config.COHORT_MODEL_SENTENCES_DIR}")


def generate_gpt2_predictors(data_dir, predictor_dir, file_list=None):
    """
    Generate predictors for GPT-2 model features
    
    Creates predictors with:
    - word_number
    - word_surprisal_GPT
    - word_entropy_GPT
    
    Parameters
    ----------
    data_dir : Path
        Directory containing GPT-2 model CSV files
    predictor_dir : Path
        Output directory for predictors
    file_list : list, optional
        List of file basenames to process. If None, process all CSV files.
    """
    print(f"\n{'='*70}")
    print("GENERATING GPT-2 MODEL PREDICTORS")
    print(f"{'='*70}")
    print(f"Data directory: {data_dir}")
    print(f"Predictor directory: {predictor_dir}")
    
    # Ensure output directory exists
    predictor_dir.mkdir(parents=True, exist_ok=True)
    
    if file_list is None:
        file_list = get_gpt_model_files(data_dir)
    
    if not file_list:
        print("No files to process")
        return
    
    print(f"Files to process: {len(file_list)}")
    
    for segment in file_list:
        csv_file = data_dir / f'{segment}_words_phonemes_revised_revised_cohort_model_GPT.csv'
        
        if not csv_file.exists():
            print(f"Warning: CSV file not found for {segment}, skipping")
            continue
        
        print(f"Processing: {segment}")
        
        # Load segment table
        segment_table = load.tsv(csv_file, delimiter=';')
        
        # Create dataset with timing information
        ds = Dataset(
            {'time': segment_table['phoneme_onset']}, 
            info={'tstop': segment_table[-1, 'phoneme_offset']}
        )
        
        # Add predictor variables
        for key in ['word_number', 'word_surprisal_GPT', 'word_entropy_GPT']:
            ds[key] = segment_table[key]
        
        # Save
        output_file = predictor_dir / f'{segment}~phoneme_cohort_model_GPT2_new_large.pickle'
        save.pickle(ds, output_file)
    
    print(f"\nCompleted processing {len(file_list)} files")


def generate_all_gpt2_predictors():
    """Generate GPT-2 model predictors for both Word_list and Sentences conditions"""
    config = Config()
    
    print(f"\n{'='*70}")
    print("GENERATING GPT-2 PREDICTORS FOR ALL CONDITIONS")
    print(f"{'='*70}")
    
    # Word_list condition
    print("\n--- Processing Word_list condition ---")
    generate_gpt2_predictors(
        data_dir=config.GPT2_MODEL_DIR,
        predictor_dir=config.WORD_PREDICTOR_DIR
    )
    
    # Sentences condition
    print("\n--- Processing Sentences condition ---")
    if config.GPT2_SENTENCES_DIR.exists():
        generate_gpt2_predictors(
            data_dir=config.GPT2_SENTENCES_DIR,
            predictor_dir=config.SENTENCE_PREDICTOR_DIR
        )
    else:
        print(f"Warning: Sentences directory not found: {config.GPT2_SENTENCES_DIR}")


# =============================================================================
# Main execution
# =============================================================================

def main():
    """
    Main execution function
    
    Uncomment the sections you want to run.
    """
    config = Config()
    
    print("="*70)
    print("PREDICTOR GENERATION PIPELINE")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # SECTION 1: Generate gammatone spectrograms for word lists
    # -------------------------------------------------------------------------
    generate_gammatone_spectrograms(config.WORD_LIST_STIMULI)
    
    # -------------------------------------------------------------------------
    # SECTION 2: Generate gammatone spectrograms for sentences
    # -------------------------------------------------------------------------
    generate_gammatone_spectrograms(config.SENTENCE_STIMULI)
    
    # -------------------------------------------------------------------------
    # SECTION 3: Generate gammatone-based predictors for BOTH conditions
    # -------------------------------------------------------------------------
    generate_all_gammatone_predictors()
    
    # -------------------------------------------------------------------------
    # SECTION 4: Generate cohort model predictors for BOTH conditions
    # -------------------------------------------------------------------------
    generate_all_cohort_model_predictors()
    
    # -------------------------------------------------------------------------
    # SECTION 5: Generate GPT-2 model predictors for BOTH conditions
    # -------------------------------------------------------------------------
    generate_all_gpt2_predictors()
    
    print("\n" + "="*70)
    print("All sections are currently commented out.")
    print("Uncomment the sections you want to run in the main() function.")
    print("="*70)


if __name__ == '__main__':
    main()