"""
Generate predictors for word-level variables

This script creates word-level linguistic predictors from cohort model data,
including cohort entropy and cohort surprisal aligned to phoneme onsets.
"""
from pathlib import Path
import os
import eelbrain


# Paths
root = Path.cwd().parents[0]
DATA_ROOT = root / 'Materials'

# Dutch_stimuli or Chinese_stimuli; OR Dutch_participants or Chinese_participants
# Words or Syllables

STIMULUS_DIR = DATA_ROOT / 'Dutch_stimuli' / 'Cohort_model' / 'Dutch_participants' / 'Words' 
PREDICTOR_DIR = DATA_ROOT /  'Dutch_stimuli' / 'Predictors' / 'Dutch_participants' / 'Words'

# Predictor variables to extract
PREDICTOR_VARS = ['cohort_entropy', 'cohort_surprisal']

# Ensure output directory exists
PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)

# Get list of cohort model files
list_wav = [f.split('transcription')[0] for f in os.listdir(STIMULUS_DIR) 
            if f.endswith('_cohort_model.csv')]


def generate_word_predictors(segment):
    """Generate word-level predictors for a single audio segment"""
    # Construct output filename
    name = 'Audio_nouns_' + segment.split('_')[2] + '_' + segment.split('_')[3]
    dst = PREDICTOR_DIR / f'{name}~phoneme_cohort_model.pickle'
    
    if dst.exists():
        return True, name
    
    print(f"Processing: {name}")
    
    # Load cohort model data
    segment_table = eelbrain.load.tsv(
        STIMULUS_DIR / f'{segment}transcription_cohort_model.csv',
        delimiter=';'
    )
    
    # Create dataset with time information
    ds = eelbrain.Dataset(
        {'time': segment_table['phoneme_onset']}, 
        info={'tstop': segment_table[-1, 'phoneme_offset']}
    )
    
    # Add predictor variables
    for key in PREDICTOR_VARS:
        ds[key] = segment_table[key]
    
    # Save dataset
    eelbrain.save.pickle(ds, dst)
    
    return False, name


def main():
    """Run the word predictor generation pipeline"""
    print(f"Processing {len(list_wav)} word segments...")
    print(f"Output directory: {PREDICTOR_DIR}")
    print(f"Predictor variables: {', '.join(PREDICTOR_VARS)}")
    print()
    
    print("=" * 60)
    print("Generating word-level predictors")
    print("=" * 60)
    
    skipped = 0
    for segment in list_wav:
        existed, name = generate_word_predictors(segment)
        if existed:
            skipped += 1
    
    print()
    print(f"Completed: {len(list_wav) - skipped} generated, {skipped} skipped (already exist)")
    print()
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()