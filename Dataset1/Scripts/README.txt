================================================================================
MEG ANALYSIS PIPELINE FOR TEMPORAL RESPONSE FUNCTION (TRF) STUDIES
================================================================================

Author: Filiz Tezcan
Institution: Max Planck Institute for Psycholinguistics
Project: Linguistic structure and language familiarity sharpen phoneme encoding in the brain.

================================================================================
OVERVIEW
================================================================================

This pipeline processes MEG data to analyze how the brain encodes hierarchical
linguistic information during speech processing. The analysis compares neural
responses to sentences versus isolated words using Temporal Response Functions
(TRFs) with multiple predictor features:

- Acoustic edges (gammatone filterbank onsets)
- Phoneme onsets
- Phoneme surprisal (cohort model)
- Phoneme entropy (cohort model)
- Word-level surprisal (GPT-2)
- Word-level entropy (GPT-2)

The pipeline supports analysis of MEG data filtered to delta and theta
frequency bands, focusing on Superior Temporal Gyrus (STG) sources.

================================================================================
PIPELINE COMPONENTS
================================================================================

1. PREPROCESSING & FEATURE PREPARATION
   - Preprocess_MEG_data.py: MEG preprocessing with ICA
   - make_predictors.py: Generate TRF predictor features
   - Cohort_model.py: Calculate phoneme-level surprisal/entropy
   - Add_GPT_features.py: Add word-level surprisal/entropy from GPT-2

2. TRF ESTIMATION
   - estimate_trfs_unified.py: Fit TRF models to MEG data

3. ANALYSIS & VISUALIZATION
   - TRF_weight_analysis.py: Compare TRF weights between conditions
   - Accuracy_analysis.py: Compare feature contributions
   - Accuracy_analysis_response_letter.py: Phoneme onset vs surprisal/entropy
   - Accuracy_analysis_response_letter2.py: Shuffled vs normal phoneme features
   - visualize_trf_results.py: Generate figures

================================================================================
DIRECTORY STRUCTURE
================================================================================

Project Root/
├── raw/                           # Raw MEG data (.ds files)
│   └── sub-XXX/
│       └── ses-megXX/
│           └── meg/
├── processed/                     # Processed MEG data
│   └── sub-XXX/
│       └── meg/
│           ├── *-raw.fif          # Preprocessed continuous data
│           ├── *-eve.fif          # Event markers
│           ├── TRF/               # TRF results
│           │   ├── Sentences/
│           │   └── Word_list/
├── Materials/                     # Linguistic materials
│   ├── Dutch_dict_2022.txt        # Grapheme-to-phoneme dictionary
│   ├── SUBTLEX-NL_filtered_2022_cut.csv  # Word frequencies
│   ├── Stimuli/
│   │   └── Transcription/
│   │       ├── Word_phoneme_transcription_of_sentences/
│   │       └── Word_phoneme_transcription_of_words/
│   ├── Cohort_model/              # Cohort model outputs
│   │   ├── Sentences/
│   │   └── Word_list/
│   └── GPT2/                      # GPT-2 outputs
│       ├── Sentences/
│       └── Word_list/
└── Scripts/                       # Analysis scripts (this repository)
    ├── Accuracy_analysis/
    │   └── Results_publication/
    ├── TRF_weight_analysis/
    │   └── Output/
    └── Visualize_TRF_weights/
        └── Output/



================================================================================
DATA REQUIREMENTS
================================================================================

INPUT DATA:
1. Raw MEG data in CTF format (.ds directories)
2. Phoneme transcriptions (CSV files with words and phonemes)
3. Dutch grapheme-to-phoneme dictionary
4. SUBTLEX-NL word frequency norms
5. Audio stimuli (WAV files for acoustic feature extraction)

OUTPUT DATA:
1. Preprocessed MEG: *-raw.fif files (300 Hz, ICA-cleaned)
2. Events: *-eve.fif files (stimulus and response triggers)
3. TRF models: *.pickle files (eelbrain format)
4. Statistical results: *.npy and *.csv files
5. Figures: *.svg files (publication-ready)

================================================================================
ANALYSIS WORKFLOW
================================================================================

STAGE 1: DATA PREPARATION
   Step 1: Preprocess MEG data (Preprocess_MEG_data.py)
           - Filter, resample, ICA artifact removal
           - Manual component selection required
   
   Step 2: Calculate cohort model features (Cohort_model.py)
           - Phoneme-level surprisal and entropy
   
   Step 3: Add GPT-2 features (Add_GPT_features.py)
           - Word-level surprisal and entropy
   
   Step 4: Create TRF predictors (make_predictors.py)
           - Acoustic edges from audio
           - Combine all linguistic features
           - Normalize and time-align

STAGE 2: TRF ESTIMATION
   Step 5: Fit TRF models (estimate_trfs_unified.py)
           - Multiple nested models per subject
           - Separate for Sentences and Word_list
           - Both hemispheres (lh, rh)

STAGE 3: STATISTICAL ANALYSIS
   Step 6: Compare TRF weights (TRF_weight_analysis.py)
           - Cluster-based permutation tests
           - Sentences vs Word_list
   
   Step 7: Compare model accuracies (Accuracy_analysis*.py)
           - Feature contribution analysis
           - Mixed-effects modeling preparation

STAGE 4: VISUALIZATION
   Step 8: Generate figures (visualize_trf_results.py)
           - Time-course plots
           - Statistical overlays
           - Publication-quality output

================================================================================
ANALYSIS CONDITIONS
================================================================================

EXPERIMENTAL CONDITIONS:
- Sentences: Continuous spoken Dutch sentences (natural speech)
- Word_list: Isolated Dutch words (no sentence context)

TRF MODELS (nested hierarchy):
1. Baseline: Acoustic edges only
2. +Phoneme onset: Adds phoneme timing
3. +Phoneme surp/ent: Adds cohort model predictions
4. +Words: Adds word-level features (full model)

SUBJECT GROUPS:
- Native Dutch speakers (n=19)
- Subjects: sub-002 through sub-021 (excluding sub-015)

ROI ANALYSIS:
- Superior Temporal Gyrus (STG) sources
- Left and right hemispheres analyzed separately
- Delta (1-4 Hz) and theta (4-8 Hz) frequency bands

================================================================================
SCRIPT DESCRIPTIONS
================================================================================

Preprocess_MEG_data.py
----------------------
Preprocesses raw MEG data with manual ICA component selection.
- Loads CTF format data
- Resamples to 300 Hz
- Applies 1-100 Hz bandpass filter
- Fits ICA and displays components for manual inspection
- REQUIRES MANUAL INTERVENTION: User must specify bad components
- Saves cleaned data and events

INPUT: Raw MEG .ds directories
OUTPUT: *-raw.fif (preprocessed), *-eve.fif (events)

make_predictors.py
------------------
Generates TRF predictor matrices from audio and linguistic features.
- Extracts acoustic edges using gammatone filterbank
- Loads phoneme transcriptions with cohort model features
- Loads GPT-2 word-level features
- Creates time-aligned predictor matrices
- Normalizes features

INPUT: Audio files, phoneme transcriptions, cohort/GPT features
OUTPUT: Predictor pickle files for TRF estimation

Cohort_model.py
---------------
Calculates phoneme surprisal and entropy using cohort model.
- Implements incremental word recognition model
- Uses Dutch pronunciation dictionary
- Uses SUBTLEX-NL word frequencies
- Computes surprisal and entropy at each phoneme position

INPUT: Phoneme transcriptions, dictionary, word frequencies
OUTPUT: CSV files with cohort model features per phoneme

Add_GPT_features.py
-------------------
Adds word-level surprisal and entropy from GPT-2.
- Uses Dutch GPT-2 language model (yhavinga/gpt2-large-dutch)
- Calculates surprisal for each word given previous context
- Computes entropy over vocabulary at word onset
- Assigns features to first phoneme of each word

INPUT: Phoneme transcriptions with cohort features
OUTPUT: CSV files with added GPT-2 features

estimate_trfs_unified.py
------------------------
Estimates TRF models relating predictors to MEG responses.
- Fits multiple nested models per subject
- Uses boosting for regularization
- Processes both conditions and hemispheres
- Supports SLURM cluster parallelization

INPUT: Preprocessed MEG, predictor matrices
OUTPUT: TRF model objects (pickle files)

TRF_weight_analysis.py
----------------------
Compares TRF weights between conditions using cluster statistics.
- Computes RMS of weights across sources
- Performs cluster-based permutation tests
- Tests Sentences vs Word_list differences
- Identifies significant temporal clusters

INPUT: TRF pickle files for both conditions
OUTPUT: Cluster statistics (pickle), significance masks (npy)

Accuracy_analysis.py
--------------------
Analyzes feature contributions by comparing nested models.
- Computes explained variance (R²) for each model
- Calculates feature contribution as difference in R²
- Compares acoustic edge vs phoneme features
- Generates boxplots with statistics

INPUT: TRF pickle files (multiple nested models)
OUTPUT: CSV data, boxplot figures (SVG)

Accuracy_analysis_response_letter.py
------------------------------------
Compares phoneme onset vs surprisal/entropy contributions.
- Separates phoneme timing from prediction features
- Tests which component drives neural responses
- Addresses reviewer question about feature decomposition

INPUT: TRF pickle files (phoneme feature variants)
OUTPUT: CSV data, comparison figures (SVG)

Accuracy_analysis_response_letter2.py
-------------------------------------
Tests incremental vs shuffled phoneme processing.
- Compares normal word-by-word processing
- Against shuffled phoneme order (non-incremental)
- Tests whether incrementality matters for prediction

INPUT: TRF pickle files (normal vs shuffled)
OUTPUT: CSV data, comparison figures (SVG)

visualize_trf_results.py
------------------------
Creates publication-quality visualizations of TRF results.
- Plots time-courses of TRF weight power
- Shows Sentences vs Word_list comparison
- Overlays statistical significance
- Generates multi-panel figures

INPUT: TRF pickle files, cluster statistics
OUTPUT: Multi-panel SVG figures

================================================================================
CONFIGURATION NOTES
================================================================================

PATHS:
- All scripts use relative paths from a common project root
- Modify root = Path.cwd().parents[N] as needed for your structure
- SUBJECTS_DIR should point to your processed MEG directory

SUBJECTS:
- Subject list defined in each analysis script
- Exclude subjects with poor data quality as needed
- Subject naming: 'sub-XXX' format

PARAMETERS:
- TRF time window: typically -0.1 to 0.6 seconds
- Sampling rate: 300 Hz after preprocessing
- Frequency bands: Delta (1-4 Hz), Theta (4-8 Hz)
- Number of sources: depends on parcellation (STG ROI)

MODELS:
- Model names indicate features: "Control2_Delta+Theta_STG_sources_normalized_..."
- Cohort model uses SUBTLEX-NL frequencies
- GPT-2 model: "yhavinga/gpt2-large-dutch" (change if needed)


================================================================================
CITATION 
================================================================================

If you use this pipeline, please cite:

Tezcan, F., Ten Oever, S., Bai, F., te Rietmolen, N., & Martin, A. (2025). 
Linguistic structure and language familiarity sharpen phoneme encoding in the brain.


================================================================================
CONTACT & SUPPORT
================================================================================

Author: Filiz Tezcan
Email: filiz.tezcansemerci@mpi.nl
Lab: MPI for Psycholinguistics
