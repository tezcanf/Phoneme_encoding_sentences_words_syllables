================================================================================
UNIFIED TRF ESTIMATION PIPELINE
================================================================================

Author: Filiz Tezcan
Institution: Max Planck Institute for Psycholinguistics
Project: Linguistic structure and language familiarity sharpen phoneme encoding in the brain.

================================================================================
OVERVIEW
================================================================================

This unified pipeline estimates Temporal Response Functions (TRFs) for MEG data
analyzing how the brain encodes speech across different languages and conditions.
The pipeline has been streamlined to handle multiple experimental configurations:

KEY FEATURES:
- Dutch and Chinese participant groups
- Native and non-native stimulus processing
- Multiple conditions (Words, Syllables)
- Flexible configuration system
- SLURM cluster support
- Validation tools to check data integrity

EXPERIMENTAL DESIGN:
- Dutch subjects: Listen to Dutch stimuli (native) OR Chinese stimuli (non-native)
- Chinese subjects: Listen to Chinese stimuli (native) OR Dutch stimuli (non-native)
- Conditions: Words OR Syllables

TRF MODELS:
1. Acoustic only: Gammatone spectrogram + acoustic edges
2. Acoustic + phonemes: Adds phoneme onsets, surprisal, entropy
3. Phoneme only: Phonemes without acoustic edges
4. Phoneme decomposed: Separate surprisal/entropy and onset models
5. Phoneme onset only: Just timing information

================================================================================
PIPELINE COMPONENTS
================================================================================

CORE SCRIPTS:
-------------
1. config.py
   - Central configuration file
   - Defines subject lists (Dutch and Chinese)
   - Sets data paths
   - Defines TRF parameters
   - Manages model specifications

2. utils_unified.py
   - Unified utility functions
   - Source space reconstruction
   - Handles both Dutch and Chinese stimuli
   - Creates MEG source estimates in STG ROI

3. estimate_trfs_unified.py
   - Main TRF estimation script
   - Command-line interface
   - Processes multiple subjects/conditions
   - Supports native/non-native stimulus configurations

INTERACTIVE RUNNERS:
-------------------
4. run_trf_estimation.py
   - Jupyter/IPython friendly version
   - No command-line arguments needed
   - Edit variables at top of script

5. run_batch.py
   - Simplified batch processing
   - Edit settings and run
   - Easier than command-line arguments

VALIDATION & UTILITIES:
-----------------------
6. validate_setup.py
   - Pre-flight checks before TRF estimation
   - Verifies all required files exist
   - Checks directory structure
   - Reports missing files

PREDICTOR GENERATION:
--------------------
7. Make_gammatone.py
   - Generates acoustic features
   - Creates gammatone spectrograms
   - Detects acoustic onsets
   - Frequency binning

8. Make_phoneme_predictors.py
   - Creates phoneme-level predictors
   - Loads cohort model outputs
   - Aligns to audio timeline

PREPROCESSING:
-------------
9. Preprocess_MEG_data.py
   - MEG preprocessing pipeline
   - ICA artifact removal
   - Filtering and resampling
   - Event extraction

ANALYSIS SCRIPTS:
----------------
10. TRF_weight_analysis.py
    - Merged analysis script
    - Language comparisons (Dutch vs Chinese stimuli)
    - Condition comparisons (Words vs Syllables)
    - Participant group comparisons
    - Statistical testing

11. Accuracy_analysis.py
    - Model comparison analysis
    - Feature contribution testing
    - Statistical reporting

12. Accuracy_analysis_response_letter.py
    - Reviewer response analysis
    - Phoneme onset vs surprisal/entropy decomposition

13. visualize_trf_results.py
    - Publication figure generation
    - Time-course plots

COHORT MODEL:
------------
14. Cohort_model_Dutch.py
    - Phoneme-level surprisal and entropy for Dutch stimuli
    - Uses Dutch pronunciation dictionary (Dutch_dict_2022.txt)
    - Dutch word frequencies (SUBTLEX-NL_filtered_2022_cut.csv)
    - Outputs: *_cohort_model.csv files

15. Cohort_model_Chinese.py
    - Phoneme-level surprisal and entropy for Chinese stimuli
    - Uses Chinese pronunciation dictionary (Chinese_dict.txt)
    - Chinese word frequencies (Chinese_freq_file.xlsx)
    - Outputs: *_cohort_model.csv files

NOTE: These scripts are separate (not merged) and configured via inline variables:
- Condition: 'Words' or 'Syllables'
- DATASET: 'Dutch_participants' or 'Chinese_participants'
- stimuli: 'Dutch_stimuli' or 'Chinese_stimuli'

================================================================================
DIRECTORY STRUCTURE
================================================================================

Project Root/
├── Materials/
│   ├── Dutch_stimuli/                           # All Dutch stimulus materials
│   │   ├── Stimuli/
│   │   │   ├── Dutch_participants/
│   │   │   │   ├── Words/
│   │   │   │   │   └── Audio_*.wav
│   │   │   │   └── Syllables/
│   │   │   │       └── Audio_*.wav
│   │   │   └── Chinese_participants/
│   │   │       ├── Words/
│   │   │       └── Syllables/
│   │   ├── Predictors/
│   │   │   ├── Dutch_participants/
│   │   │   │   ├── Words/
│   │   │   │   │   ├── *~gammatone-8.pickle
│   │   │   │   │   ├── *~gammatone-on-8.pickle
│   │   │   │   │   └── *~phoneme_cohort_model.pickle
│   │   │   │   └── Syllables/
│   │   │   └── Chinese_participants/
│   │   │       ├── Words/
│   │   │       └── Syllables/
│   │   ├── Cohort_model_Dutch_participants/
│   │   │   ├── Words/
│   │   │   │   └── *_transcription_cohort_model.csv
│   │   │   └── Syllables/
│   │   ├── Cohort_model_Chinese_participants/
│   │   │   ├── Words/
│   │   │   └── Syllables/
│   │   ├── Dutch_dict_2022.txt                  # G2P dictionary
│   │   ├── SUBTLEX-NL_filtered_2022_cut.csv     # Frequency data
│   │   └── phone2int_cohort.pkl                 # Phoneme mappings
│   │
│   └── Chinese_stimuli/                         # All Chinese stimulus materials
│       ├── Stimuli/
│       │   ├── Dutch_participants/
│       │   │   ├── Words/
│       │   │   └── Syllables/
│       │   └── Chinese_participants/
│       │       ├── Words/
│       │       └── Syllables/
│       ├── Predictors/
│       │   ├── Dutch_participants/
│       │   │   ├── Words/
│       │   │   └── Syllables/
│       │   └── Chinese_participants/
│       │       ├── Words/
│       │       └── Syllables/
│       ├── Cohort_model_Dutch_participants/
│       ├── Cohort_model_Chinese_participants/
│       ├── Chinese_dict.txt
│       └── Chinese_freq_file.xlsx
│
├── Dutch_participants/                          # Dutch subject MEG data
│   ├── raw/
│   │   └── sub-003/
│   │       └── ses-meg02/
│   │           └── meg/
│   │               └── *.ds (CTF format)
│   └── processed/
│       └── sub-003/
│           ├── bem/                             # FreeSurfer BEM
│           ├── mri/                             # FreeSurfer MRI
│           ├── surf/                            # FreeSurfer surfaces
│           └── meg/
│               ├── sub-003_resampled_300Hz-1-100Hz_filtered-ICA-eyeblink_Dutch_stimuli.fif
│               ├── sub-003_resampled_300Hz-1-100Hz_filtered-ICA-eyeblink_Chinese_stimuli.fif
│               ├── sub-003_forward.fif
│               ├── sub-003_inverse.fif
│               ├── sub-003-trans.fif
│               ├── sub-003-eve_Dutch_stimuli.fif
│               ├── sub-003-eve_Chinese_stimuli.fif
│               └── TRF/
│                   ├── Words/
│                   │   ├── sub-003 Control2_*_Dutch_stimuli_acoustic_lh.pickle
│                   │   ├── sub-003 Control2_*_Dutch_stimuli_acoustic_rh.pickle
│                   │   ├── sub-003 Control2_*_Chinese_stimuli_acoustic_lh.pickle
│                   │   └── sub-003 Control2_*_Chinese_stimuli_acoustic_rh.pickle
│                   └── Syllables/
│
└── Chinese_participants/                        # Chinese subject MEG data
    ├── raw/
    │   └── sub-021/
    └── processed/
        └── sub-021/
            ├── bem/
            ├── mri/
            ├── surf/
            └── meg/
                └── (similar structure to Dutch subjects)

Scripts/                                          # Analysis scripts (your working directory)
├── config.py
├── utils_unified.py
├── estimate_trfs_unified.py
├── run_trf_estimation.py
├── run_batch.py
├── validate_setup.py
├── TRF_weight_analysis.py
│   └── Output/
│       ├── clu_*.pickle
│       └── *.npy
├── Accuracy_analysis.py
└── visualize_trf_results.py

================================================================================
DATA REQUIREMENTS
================================================================================

INPUT FILES:
-----------
1. Preprocessed MEG Data:
   Format: *_resampled_300Hz-1-100Hz_filtered-ICA-eyeblink_{stimulus_type}.fif
   - Dutch subjects: Need both Dutch_stimuli.fif and Chinese_stimuli.fif versions
   - Chinese subjects: Need both Chinese_stimuli.fif and Dutch_stimuli.fif versions
   - Sampling rate: 300 Hz
   - Frequency range: 1-100 Hz
   - ICA cleaned

2. Forward/Inverse Solutions:
   - {subject}_forward.fif: Forward solution for source localization
   - {subject}_inverse.fif: Inverse operator for source reconstruction

3. Events Files:
   Format: {subject}-eve_{stimulus_type}.fif
   - Event codes: 10 (Words), 20 (Syllables), 30 (Sentences)
   - Timing markers for stimulus onsets

4. FreeSurfer Anatomy:
   - Subject-specific anatomy in subjects_dir
   - Required for source space reconstruction
   - STG parcellation (aparc.a2009s)

5. Stimulus Audio:
   - WAV files for acoustic feature extraction
   - Organized by participant group and condition

6. Predictor Files:
   - Gammatone spectrograms: *~gammatone-8.pickle
   - Acoustic onsets: *~gammatone-on-8.pickle
   - Phoneme features: *~phoneme_cohort_model.pickle
   - Must match stimulus audio files

OUTPUT FILES:
------------
1. TRF Models:
   Format: {subject} {model_name}_{hemisphere}.pickle
   - Contains fitted TRF weights
   - Model performance metrics
   - Cross-validation results
   - Separate for each hemisphere (lh, rh)

2. Analysis Results:
   - Cluster statistics: clu_*.pickle
   - Significance masks: *.npy
   - Accuracy comparisons: *.csv
   - Publication figures: *.svg

================================================================================
SUBJECT LISTS
================================================================================

DUTCH SUBJECTS (n=15):
sub-003, sub-005, sub-007, sub-008, sub-009, sub-010, sub-012, sub-013, 
sub-014, sub-015, sub-016, sub-017, sub-018, sub-019, sub-020

CHINESE SUBJECTS (n=14):
sub-021, sub-022, sub-023, sub-024, sub-025, sub-026, sub-027, sub-028, 
sub-029, sub-030, sub-032, sub-033, sub-034, sub-035

================================================================================
CONFIGURATION SYSTEM
================================================================================

The pipeline uses a centralized configuration in config.py:

SUBJECT GROUPS:
- DUTCH_SUBJECTS: List of Dutch participant IDs
- CHINESE_SUBJECTS: List of Chinese participant IDs

STIMULUS TYPES:
- 'native': Dutch subjects → Dutch stimuli, Chinese subjects → Chinese stimuli
- 'non-native': Dutch subjects → Chinese stimuli, Chinese subjects → Dutch stimuli
- 'Dutch': All subjects → Dutch stimuli
- 'Chinese': All subjects → Chinese stimuli

DATA PATHS (config.py):
- ROOT_DIR: Path.cwd().parents[1]  # Two levels up from Scripts/
- get_materials_root(stimulus_type): ROOT_DIR / 'Materials' / f'{stimulus_type}_stimuli'
- get_meg_root(subject_group): ROOT_DIR / f'{subject_group}_participants' / 'processed'
- get_subject_config(subject, stimulus_type): Returns complete config dict

TRF PARAMETERS (TRF_PARAMS in config.py):
- tmin: -0.05 s (pre-stimulus baseline)
- tmax: 0.7 s (post-stimulus response)
- error: 'l2' (L2 regularization)
- basis: 0.050 s (temporal basis function width)
- partitions: 5 (cross-validation folds)
- test: 1 (test on partition 1)
- selective_stopping: True (prevents overfitting)
- n_stimuli: 40 (number of trials per condition)

MODEL SPECIFICATIONS:
- get_models(stimulus_type) function returns model definitions
- Models vary by stimulus type (Dutch vs Chinese)
- Naming: Control2_Delta+Theta_STG_sources_normalized_{stimulus_type}_stimuli_{features}
- Nested hierarchy for feature contribution analysis

================================================================================
TRF ESTIMATION WORKFLOW
================================================================================

BASIC WORKFLOW:
1. Validate setup 
2. Run TRF estimation
3. Analyze results
4. Generate figures

VALIDATION BEFORE ESTIMATION:
python validate_setup.py --group all --stimulus-type native

This checks:
✓ All required directories exist
✓ Stimulus files present for each subject
✓ Predictor files exist
✓ MEG files correctly named
✓ Events files present
✓ FreeSurfer anatomy available

TRF ESTIMATION OPTIONS:

Option A: Command-line (Most Flexible)
---------------------------------------
# Single subject, native stimuli
python estimate_trfs_unified.py --subjects sub-003 --conditions Words

# All Dutch subjects with Chinese stimuli (non-native)
python estimate_trfs_unified.py --group dutch --stimulus-type Chinese

# All Chinese subjects with Dutch stimuli (non-native)
python estimate_trfs_unified.py --group chinese --stimulus-type Dutch

# All subjects, native stimuli, all conditions
python estimate_trfs_unified.py --all --stimulus-type native

# All subjects, non-native stimuli
python estimate_trfs_unified.py --all --stimulus-type non-native

Option B: Interactive Script
-----------------------------
1. Edit run_trf_estimation.py:
   SUBJECTS_TO_PROCESS = ['sub-003']
   CONDITIONS_TO_PROCESS = ['Words']
   STIMULUS_TYPE = 'native'

2. Run:
   python run_trf_estimation.py

Option C: Batch Script
----------------------
1. Edit run_batch.py:
   MODE = 'dutch'  # or 'chinese', 'all', 'single', 'multiple'
   SUBJECTS = ['sub-003']
   CONDITIONS = ['Words', 'Syllables']
   STIMULUS_TYPE = 'native'

2. Run:
   python run_batch.py

================================================================================
ANALYSIS CONFIGURATIONS
================================================================================

TRF_weight_analysis.py supports multiple analysis types:

CONFIGURATION VARIABLES:
- PARTICIPANT_GROUP: 'Dutch' or 'Chinese'
- ANALYSIS_TYPE: 'language_comparison' or 'condition_comparison'
- VERSION: 'original' or 'revision'
- STIMULI_LANGUAGE: 'Dutch' or 'Chinese' (for condition_comparison only)

ANALYSIS SCENARIOS:

1. Language Comparison (Dutch vs Chinese Stimuli):
   PARTICIPANT_GROUP = 'Chinese'
   ANALYSIS_TYPE = 'language_comparison'
   → Compares Chinese subjects listening to Dutch vs Chinese stimuli

2. Condition Comparison (Words vs Syllables):
   PARTICIPANT_GROUP = 'Dutch'
   ANALYSIS_TYPE = 'condition_comparison'
   STIMULI_LANGUAGE = 'Dutch'
   → Compares Words vs Syllables for Dutch subjects with Dutch stimuli

3. Version Control:
   VERSION = 'original': Tests acoustic edge + average linguistic features
   VERSION = 'revision': Tests phoneme onset + average surprisal/entropy features

================================================================================
COMMAND-LINE REFERENCE
================================================================================

estimate_trfs_unified.py:
------------------------
Arguments:
  --subjects SUB1 SUB2 ...    Process specific subjects
  --conditions COND1 COND2    Process specific conditions (default: all)
  --group {dutch,chinese,all} Process all subjects in a group
  --all                       Process all subjects and conditions
  --stimulus-type {native,non-native,Dutch,Chinese}
                              Stimulus language configuration

Examples:
  # Test single subject
  python estimate_trfs_unified.py --subjects sub-003 --conditions Words

  # Dutch group with native stimuli
  python estimate_trfs_unified.py --group dutch --stimulus-type native

  # Cross-linguistic: Dutch subjects with Chinese stimuli
  python estimate_trfs_unified.py --group dutch --stimulus-type Chinese

  # Everyone, everything
  python estimate_trfs_unified.py --all --stimulus-type native

validate_setup.py:
-----------------
Arguments:
  --subjects SUB1 SUB2 ...    Validate specific subjects
  --group {dutch,chinese,all} Validate all subjects in a group
  --conditions COND1 COND2    Validate specific conditions (default: all)
  --stimulus-type {native,non-native,Dutch,Chinese}
                              Stimulus type to validate
  --quick                     Quick check (only first subject per group)

Examples:
  # Quick validation
  python validate_setup.py --quick --stimulus-type native

  # Full validation for Dutch group
  python validate_setup.py --group dutch --stimulus-type native

  # Check specific subject
  python validate_setup.py --subjects sub-003 --stimulus-type native

================================================================================
CLUSTER COMPUTING
================================================================================

For large-scale processing, use SLURM batch scripts:

SINGLE SUBJECT JOB:
#!/bin/bash
#SBATCH --job-name=trf_sub003
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load Python/3.9.6
source activate trf_env

python estimate_trfs_unified.py \
    --subjects sub-003 \
    --conditions Words Syllables \
    --stimulus-type native

ARRAY JOB (All Subjects):
#!/bin/bash
#SBATCH --job-name=trf_array
#SBATCH --array=1-15  # Number of subjects
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

module load Python/3.9.6
source activate trf_env

# Subject list
SUBJECTS=(sub-003 sub-005 sub-007 sub-008 sub-009 sub-010 sub-012 sub-013 
          sub-014 sub-015 sub-016 sub-017 sub-018 sub-019 sub-020)

# Get subject for this array task
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID-1]}

python estimate_trfs_unified.py \
    --subjects $SUBJECT \
    --stimulus-type native

NOTES:
- Adjust memory based on number of sources
- Use --cpus-per-task to match boosting n_jobs parameter

================================================================================
MODEL DESCRIPTIONS
================================================================================

The pipeline estimates multiple nested models to assess feature contributions:

ACOUSTIC MODEL:
- Gammatone spectrogram (8 frequency bands)
- Acoustic edges/onsets

ACOUSTIC + PHONEMES MODEL:
- All acoustic features
- Phoneme onsets
- Phoneme surprisal (cohort model)
- Phoneme entropy (cohort model)

PHONEME ONLY MODEL:
- Gammatone spectrogram (control for spectrum)
- Phoneme onsets
- Phoneme surprisal
- Phoneme entropy
- NO acoustic edges

DECOMPOSED MODELS:
- Phoneme onset only: Just timing
- Surprisal + Entropy: Just lexical predictions

MODEL NAMING CONVENTION:
Control2_Delta+Theta_STG_sources_normalized_{stimulus_type}_stimuli_{features}

================================================================================
CITATION
================================================================================

If you use this pipeline, please cite:

Tezcan, F., Ten Oever, S., Bai, F., te Rietmolen, N., & Martin, A. E. (2025).
Linguistic structure and language familiarity sharpen phoneme encoding in the brain.

================================================================================
CONTACT & SUPPORT
================================================================================

Author: Filiz Tezcan
Email: filiz.tezcansemerci@mpi.nl
Institution: Donders Institute / Max Planck Institute for Psycholinguistics

For questions about:
- TRF methods: See eelbrain documentation (https://eelbrain.readthedocs.io/)
- MEG preprocessing: See MNE-Python documentation (https://mne.tools/)
- Configuration issues: Check config.py and validate_setup.py output
- Analysis questions: Contact author

================================================================================
