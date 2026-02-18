================================================================================
MEG TEMPORAL RESPONSE FUNCTION (TRF) ANALYSIS PIPELINE
================================================================================

Author: Filiz Tezcan
Institution: Max Planck Institute for Psycholinguistics
Project: Linguistic structure and language familiarity sharpen phoneme encoding in the brain.

================================================================================
OVERVIEW
================================================================================

This pipeline processes MEG data to analyze how the brain encodes hierarchical
linguistic information during speech processing across different levels of
linguistic structure: sentences, isolated words, and random syllables.

The analysis uses Temporal Response Functions (TRFs) with multiple predictor
features at different linguistic levels:

ACOUSTIC FEATURES:
- Gammatone spectrogram (8-band frequency representation)
- Acoustic edges (gammatone onset detector)

PHONEME-LEVEL FEATURES:
- Phoneme onsets (timing of phoneme boundaries)
- Phoneme surprisal (cohort model predictions)
- Phoneme entropy (cohort model uncertainty)

WORD-LEVEL FEATURES (for sentences and words conditions):
- Word surprisal (GPT-2 language model predictions)
- Word entropy (GPT-2 uncertainty)

The pipeline focuses on Superior Temporal Gyrus (STG) sources in both
hemispheres and compares neural responses across three experimental conditions
to understand how linguistic hierarchy affects speech encoding.

================================================================================
KEY FEATURES
================================================================================

✓ Modular architecture with separated concerns (config, data loading, models)
✓ Support for three experimental conditions (sentences, words, syllables)
✓ Automated predictor generation from audio and linguistic annotations
✓ Source localization to STG using MNE-Python
✓ TRF estimation using eelbrain boosting algorithm
✓ Statistical comparison using cluster-based permutation tests
✓ Visualization
✓ Batch processing support for multiple subjects/conditions

================================================================================
PIPELINE COMPONENTS
================================================================================

CORE MODULES:
-------------
config.py                      - Central configuration (paths, parameters)
data_loader.py                 - MEG data loading and epoch filtering
predictors.py                  - Predictor loading and preprocessing
models.py                      - TRF model definitions
utils.py                       - Source localization utilities
trf_estimation.py              - TRF fitting functions

MAIN SCRIPTS:
-------------
estimate_trfs_pipeline.py      - Main TRF estimation pipeline (single subject)
batch_estimate_trfs.py         - Batch processing wrapper

FEATURE GENERATION:
-------------------
Cohort_model.py               - Phoneme-level surprisal/entropy (cohort model)
Add_GPT_features.py           - Word-level surprisal/entropy (GPT-2)
generate_predictors.py        - Generate all TRF predictors from audio

PREPROCESSING:
--------------
Preprocess_MEG_data.py        - MEG preprocessing with ICA 

ANALYSIS & VISUALIZATION:
-------------------------
TRF_weight_analysis.py        - Statistical comparison of TRF weights
Accuracy_analysis.py          - Feature contribution analysis
visualize_trf_results.py      - Generate publication figures

================================================================================
DIRECTORY STRUCTURE
================================================================================

Project Root/
├── processed/                        # Processed MEG data
│   └── sub-XXX/
│       └── meg/
│           ├── *_resampled_300Hz-05-145Hz_filtered-ICA-eyeblink-epochs.fif
│           ├── *-eve.fif
│           ├── *_forward_ICA_all.fif
│           ├── *_inverse_ICA_all.fif
│           └── meg/
│               └── *-trans.fif
├── Materials/                        # Linguistic materials and predictors
│   ├── Stimuli/
│   │   ├── Sounds_Syllables/
│   │   │   ├── control_sentence/    # Sentence audio files (.wav)
│   │   │   ├── random_word_list/    # Word list audio files (.wav)
│   │   │   └── random_syllables/    # Syllable audio files (.wav)
│   │   └── Block_order/
│   │       ├── Block_1.xlsx          # Stimulus presentation order
│   │       └── subjects_block_no.xlsx
│   └── Predictors/                   # Generated predictor files
│       ├── control_sentence/
│       ├── random_word_list/
│       └── random_syllables/
└── TRF_models/                       # TRF estimation outputs
    ├── control_sentence/
    │   └── sub-XXX/
    │       ├── *_lh.pickle          # Left hemisphere TRFs
    │       └── *_rh.pickle          # Right hemisphere TRFs
    ├── random_word_list/
    └── random_syllables/

================================================================================
DATA REQUIREMENTS
================================================================================

INPUT DATA:
-----------
1. Preprocessed MEG epochs:
   - Resampled to 300 Hz
   - Bandpass filtered (0.5-145 Hz)
   - ICA artifact removal applied
   - Format: MNE epochs (.fif)

2. Event markers:
   - Stimulus onset triggers
   - Format: MNE events (.fif)

3. Phoneme transcriptions:
   - Word and phoneme boundaries
   - Format: CSV with columns: 'words', 'phonemes', 'phoneme_onset', 'phoneme_offset'

4. Audio stimuli:
   - WAV files matching transcriptions
   - Used for acoustic feature extraction

5. FreeSurfer anatomical data:
   - Subject MRI reconstructions
   - BEM surfaces
   - Transformation files (*-trans.fif)

OUTPUT DATA:
------------
1. Predictor files (*.pickle):
   - Gammatone spectrograms
   - Gammatone onsets
   - Phoneme features (cohort model)
   - Word features (GPT-2)

2. TRF models (*.pickle):
   - Boosting model objects
   - Separate for each hemisphere
   - Multiple models per subject/condition

3. Statistical results:
   - Cluster permutation test results (*.pickle)
   - Significance masks (*.npy)
   - Summary statistics (*.csv)

4. Figures (*.svg):
   - Publication-quality vector graphics
   - TRF weight time courses
   - Condition comparisons

================================================================================
EXPERIMENTAL CONDITIONS
================================================================================

CONDITIONS:
-----------
sentences  (formerly 'control_sentence')
   - Full syntactic and semantic structure
   - Predictors: acoustic + phonemes + words

words  (formerly 'random_word_list')
   - No sentence context
   - Predictors: acoustic + phonemes + words

syllables  (formerly 'random_syllables')
   - No lexical or syntactic structure
   - Predictors: acoustic + phonemes only

TRF MODELS (Nested Hierarchy):
-------------------------------
For sentences and words conditions:
1. Acoustic only
   - Gammatone spectrogram + acoustic edges

2. Acoustic + phoneme onsets
   - Adds phoneme boundary timing

3. Acoustic + phoneme features
   - Adds cohort surprisal and entropy

4. Acoustic + words
   - Adds word-level surprisal and entropy

5. Full model (acoustic + phonemes + words)
   - All predictors combined

For syllables condition:
1. Acoustic only
2. Acoustic + phoneme onsets
3. Acoustic + phoneme features
4. Phonemes only (no acoustic)

SUBJECTS:
---------
- N = 30 subjects (sub-006 through sub-038, excluding sub-014, sub-029)
- Native speakers of the target language
- Right-handed
- Normal hearing

ROI ANALYSIS:
-------------
- Superior Temporal Gyrus (STG) bilateral
- Source localization using dSPM
- Morphed to fsaverage template
- Analysis in source space

================================================================================
ANALYSIS WORKFLOW
================================================================================

STAGE 1: FEATURE GENERATION
----------------------------
Step 1: Calculate cohort model features
   Script: Cohort_model.py
   Input:  Phoneme transcriptions
   Output: CSV files with phoneme surprisal and entropy

Step 2: Add GPT-2 word features
   Script: Add_GPT_features.py
   Input:  Phoneme transcriptions with cohort features
   Output: CSV files with word surprisal and entropy

Step 3: Generate TRF predictors
   Script: generate_predictors.py
   Input:  Audio files, feature CSVs
   Output: Predictor pickle files (gammatone, phonemes, words)

STAGE 2: TRF ESTIMATION
------------------------
Step 4: Estimate TRF models (single subject)
   Script: estimate_trfs_pipeline.py
   Input:  MEG epochs, predictors
   Output: TRF model pickle files (left & right hemispheres)
   
   Example:
   python estimate_trfs_pipeline.py --condition sentences --subject-index 0

Step 5: Batch processing (all subjects)
   Script: batch_estimate_trfs.py
   Input:  MEG epochs, predictors
   Output: TRF models for all subjects
   
   Example:
   python batch_estimate_trfs.py --condition sentences --all-subjects

STAGE 3: STATISTICAL ANALYSIS
------------------------------
Step 6: Compare TRF weights across conditions
   Script: TRF_weight_analysis.py
   Input:  TRF models for all subjects
   Output: Cluster statistics, significance masks
   
   Comparisons:
   - Sentences vs Words
   - Words vs Syllables

Step 7: Analyze feature contributions
   Script: Accuracy_analysis.py
   Input:  TRF models (nested hierarchy)
   Output: R² improvements, statistical tests, boxplots

STAGE 4: VISUALIZATION
-----------------------
Step 8: Generate publication figures
   Script: visualize_trf_results.py
   Input:  TRF models, statistical results
   Output: SVG figures with time courses and statistics

================================================================================
CONFIGURATION
================================================================================

PATHS (config.py):
------------------
All paths are configured relative to DATA_ROOT:
- DATA_ROOT: Project root directory (auto-detected from script location)
- STIMULUS_BASE_DIR: Audio stimuli location
- PREDICTOR_BASE_DIR: Generated predictors
- TRF_BASE_DIR: TRF model outputs

To modify paths, edit config.py:
```python
DATA_ROOT = Path('/your/custom/path')  # Or use Path.cwd().parents[1]
```

SUBJECTS (config.py):
---------------------
Subject list is defined in SUBJECTS variable:
```python
SUBJECTS = [
    'sub-006', 'sub-008', 'sub-009', ...
]
```

Add or remove subjects as needed for your dataset.

CONDITIONS (config.py):
-----------------------
Simplified condition names with backward compatibility:
```python
CONDITIONS = ['sentences', 'words', 'syllables']

CONDITION_DIR_MAPPING = {
    'sentences': 'control_sentence',    # Directory name
    'words': 'random_word_list',
    'syllables': 'random_syllables',
}
```

TRF PARAMETERS (config.py):
----------------------------
```python
TRF_PARAMS = {
    'tmin': -0.05,           # Start of TRF window (seconds)
    'tmax': 0.7,             # End of TRF window (seconds)
    'error': 'l2',           # Error function
    'basis': 0.050,          # Basis function width (seconds)
    'partitions': 5,         # Cross-validation folds
    'test': 1,               # Test partition
    'selective_stopping': True,  # Stop boosting per predictor
}
```

SOURCE LOCALIZATION (config.py):
---------------------------------
```python
SOURCE_PARAMS = {
    'crop_tmin': 1.55,           # Epoch start (seconds)
    'crop_tmax': 27.2,           # Epoch duration
    'filter_highpass': 8.0,      # High-pass filter (Hz)
    'resample_freq': 100,        # Source data sampling rate (Hz)
    'region': 'STG',             # Region of interest
}
```

================================================================================
SCRIPT DESCRIPTIONS
================================================================================

config.py
---------
Central configuration module containing all paths, parameters, and mappings.
- Defines subject list and condition names
- Sets TRF estimation parameters
- Provides helper functions for path generation
- Handles backward compatibility with legacy naming

INPUT: None (configuration only)
OUTPUT: Configuration objects used by all scripts

---

data_loader.py
--------------
Handles MEG data loading and stimulus ordering.
- Loads subject-specific stimulus presentation order
- Filters rejected epochs
- Manages block randomization
- Provides clean stimuli lists for TRF estimation

KEY FUNCTIONS:
- get_subject_stimuli(): Get stimuli in presentation order
- find_rejected_epochs(): Identify and remove bad trials
- load_and_filter_epochs(): Complete loading pipeline

INPUT: MEG epochs, events, block order files
OUTPUT: Filtered epochs, valid stimulus lists

---

predictors.py
-------------
Loads and preprocesses TRF predictors.
- Loads gammatone spectrograms
- Loads phoneme-level features (cohort model)
- Loads word-level features (GPT-2)
- Handles time alignment and resampling
- Applies padding for TRF estimation

KEY FUNCTIONS:
- load_all_predictors(): Load complete predictor set
- load_gammatone_spectrograms(): Acoustic features
- load_phoneme_predictors(): Phoneme-level features
- load_word_predictors(): Word-level features

INPUT: Predictor pickle files
OUTPUT: Time-aligned predictor matrices

---

models.py
---------
Defines TRF model specifications (which predictors to include).
- Specifies nested model hierarchy
- Condition-specific model sets
- Predictor combinations for hypothesis testing

KEY FUNCTIONS:
- get_models_for_condition(): Returns appropriate models
- get_models_with_words(): Models for sentences/words
- get_models_with_phonemes_only(): Models for syllables

INPUT: Predictor dictionary
OUTPUT: Model specifications (predictor combinations)

---

utils.py
--------
Source localization and MEG preprocessing utilities.
- Creates forward and inverse operators
- Applies inverse solution to epochs
- Morphs to fsaverage space
- Extracts STG activity
- Converts to eelbrain format

KEY FUNCTIONS:
- make_source_space(): Complete source localization pipeline
- _create_forward_solution(): Generate forward model
- _create_inverse_operator(): Generate inverse operator
- _get_region_labels(): Extract STG labels

INPUT: MEG epochs, FreeSurfer anatomy
OUTPUT: Source-space data (STG, both hemispheres)

---

trf_estimation.py
-----------------
TRF estimation core functions.
- Fits boosting TRF models
- Handles cross-validation
- Manages model saving/loading
- Processes both hemispheres

KEY FUNCTIONS:
- estimate_trfs(): Fit and save TRF models
- prepare_source_data(): Concatenate trials
- _concatenate_predictors(): Prepare predictor data

INPUT: Source data, predictors, model specifications
OUTPUT: TRF model objects (pickle files)

---

estimate_trfs_pipeline.py
--------------------------
Main pipeline script for single subject/condition TRF estimation.

USAGE:
  python estimate_trfs_pipeline.py --condition sentences --subject-index 0
  python estimate_trfs_pipeline.py --condition words --subject-index 5 --force-recompute

ARGUMENTS:
  --condition: 'sentences', 'words', or 'syllables'
  --subject-index: Integer from 0 to 30 (index into SUBJECTS list)
  --force-recompute: Overwrite existing TRF files

PIPELINE:
1. Load stimulus presentation order
2. Load and filter MEG epochs
3. Load all predictors
4. Create source space representations (STG)
5. Estimate TRFs for all models

INPUT: MEG data, predictors
OUTPUT: TRF pickle files for both hemispheres

---

batch_estimate_trfs.py
----------------------
Batch processing wrapper for multiple subjects/conditions.

USAGE:
  # All subjects, one condition
  python batch_estimate_trfs.py --condition sentences --all-subjects
  
  # Subject range
  python batch_estimate_trfs.py --condition sentences --subject-range 0 10
  
  # Multiple conditions
  python batch_estimate_trfs.py --conditions sentences words --all-subjects
  
  # Specific subjects
  python batch_estimate_trfs.py --condition sentences --subject-indices 0 1 2

FEATURES:
- Parallel processing support
- Progress tracking
- Error handling with summary
- Failed job logging

INPUT: Same as estimate_trfs_pipeline.py
OUTPUT: TRF models for all specified subjects/conditions

---

Cohort_model.py
---------------
Implements cohort model of spoken word recognition.
Calculates phoneme-level surprisal and entropy based on incremental
word recognition process.

ALGORITHM:
1. Load word frequency corpus
2. For each phoneme in sequence:
   - Maintain cohort of compatible words
   - Calculate entropy over cohort
   - Calculate surprisal of current phoneme
3. Output phoneme-level features

INPUT: Phoneme transcriptions, word frequency file, pronunciation dictionary
OUTPUT: CSV files with cohort_surprisal and cohort_entropy columns

PARAMETERS:
- max_words: Size of frequency corpus (default: 100000)
- Alphabet: Language-specific phoneme set

---

Add_GPT_features.py
-------------------
Adds word-level surprisal and entropy using GPT-2 language model.

ALGORITHM:
1. Load pre-trained GPT-2 model
2. For each word in sequence:
   - Calculate surprisal given previous context
   - Calculate entropy over vocabulary
   - Assign to first phoneme of word
3. Output augmented transcriptions

INPUT: Phoneme transcriptions with cohort features
OUTPUT: CSV files with word_surprisal_GPT and word_entropy_GPT columns

PARAMETERS:
- MODEL_NAME: HuggingFace model identifier
  Default: "ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1"
- CONTEXT_START_WORD: Initial context word

NOTES:
- Features assigned only to first phoneme of each word
- Requires GPU for faster processing (optional)
- Model downloaded automatically on first run

---

generate_predictors.py
----------------------
Generates all TRF predictors from audio and linguistic annotations.

PREDICTOR TYPES:
1. Gammatone spectrograms (high-resolution, 256 bands)
2. Gammatone 8-band predictors
3. Gammatone onset detectors
4. Phoneme-level predictors (cohort model)
5. Word-level predictors (GPT-2)

USAGE:
  python generate_predictors.py --predictor-type all
  python generate_predictors.py --predictor-type gammatone
  python generate_predictors.py --predictor-type word-gpt

OPTIONS:
  --predictor-type: Type to generate (all, gammatone, word, etc.)
  --conditions: Specific conditions to process
  --overwrite: Force regeneration of existing files

INPUT: Audio files, transcriptions, feature CSVs
OUTPUT: Predictor pickle files organized by condition

---

TRF_weight_analysis.py
----------------------
Statistical comparison of TRF weights across conditions.
Uses cluster-based permutation tests to identify temporal windows
where conditions differ significantly.

COMPARISONS:
1. Sentences vs Words
2. Words vs Syllables

FEATURES TESTED:
- Acoustic edges (gammatone onsets)
- Phoneme features (average of onset, surprisal, entropy)

ALGORITHM:
1. Load TRF models for all subjects
2. Extract and average weights across sources
3. Compute condition contrasts
4. Run cluster permutation tests (30000 permutations)
5. Extract significant time windows

INPUT: TRF pickle files for all subjects
OUTPUT: Cluster statistics (pickle), significance masks (npy)

PARAMETERS:
- P_THRESHOLD: 0.05
- N_PERMUTATIONS: 30000

---

Accuracy_analysis.py
--------------------
Analyzes feature contributions by comparing nested TRF models.
Calculates explained variance (R²) improvements when adding features.

COMPARISONS:
- Sentences vs Words (acoustic edges, phoneme features)
- Words vs Syllables (acoustic edges, phoneme features)

METRICS:
- R² improvement: Difference between nested models
- Statistical tests: One-sample t-tests against zero

OUTPUT:
- CSV files with accuracy data
- Boxplots with significance markers
- Summary statistics

VISUALIZATIONS:
- Boxplots comparing conditions
- Within-condition feature comparisons
- Statistical significance overlays

---

visualize_trf_results.py
------------------------
Generates publication-quality visualizations of TRF results.

PLOTS:
1. Time courses of TRF weight power
2. Condition comparisons (sentences vs words, words vs syllables)
3. Statistical significance overlays
4. Separate panels for acoustic and phoneme features

FEATURES:
- Customizable colors per condition
- Error bands (SEM)
- Significance markers from cluster tests
- Vector graphics output (SVG)

INPUT: TRF models, cluster statistics
OUTPUT: SVG figures

================================================================================
CITATION
================================================================================

If you use this pipeline, please cite:

Tezcan, F., Ten Oever, S., Bai, F., te Rietmolen, N., & Martin, A. (2025).
Linguistic structure and language familiarity sharpen phoneme encoding 
in the brain.

================================================================================
CONTACT & SUPPORT
================================================================================

Author: Filiz Tezcan
Email: filiz.tezcansemerci@mpi.nl
Lab: Max Planck Institute for Psycholinguistics

For questions or bug reports:
- Check documentation first
- Review example outputs
- Contact author with detailed error messages
