#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merged TRF Analysis Script
Configurable script for all TRF weight ANOVA analyses

- Chinese vs Dutch participants
- Language comparison (Dutch stimuli vs Chinese stimuli)
- Condition comparison (Words vs Syllables)
- Original vs Revision (response letter) analysis versions

@author: filiztezcan
"""

from pathlib import Path
import numpy as np
import eelbrain
import os
import pickle
from mne.stats import permutation_cluster_test

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ============================================================================

# Participant group: 'Chinese' or 'Dutch'
PARTICIPANT_GROUP = 'Chinese'

# Analysis type: 'language_comparison' or 'condition_comparison'
ANALYSIS_TYPE = 'language_comparison'

# Version: 'original' or 'revision'
VERSION = 'original'

# For condition_comparison only: 'Chinese' or 'Dutch' (which stimuli to use)
STIMULI_LANGUAGE = 'Chinese'  # Only used when ANALYSIS_TYPE == 'condition_comparison'

# ============================================================================
# DERIVED PARAMETERS - DO NOT MODIFY
# ============================================================================

DATASET = PARTICIPANT_GROUP + '_participants'
root = Path.cwd().parents[1]
subjects_dir = root / DATASET  / 'processed' 
result_folder = root / 'Scripts' / 'TRF_weight_analysis' / 'Output'


# Set up paths based on participant group
if PARTICIPANT_GROUP == 'Chinese':
    SUBJECTS = [
        'sub-021', 'sub-022', 'sub-023', 'sub-024', 'sub-025', 'sub-026',
        'sub-027', 'sub-028', 'sub-029', 'sub-030', 'sub-032', 'sub-033',
        'sub-034', 'sub-035'
    ]
elif PARTICIPANT_GROUP == 'Dutch':
    SUBJECTS = [
        'sub-003', 'sub-005', 'sub-007', 'sub-008', 'sub-009', 'sub-010',
        'sub-012', 'sub-013', 'sub-014', 'sub-015', 'sub-016', 'sub-017',
        'sub-018', 'sub-019', 'sub-020'
    ]
else:
    raise ValueError("PARTICIPANT_GROUP must be 'Chinese' or 'Dutch'")


# Set up test name and conditions
if ANALYSIS_TYPE == 'language_comparison':
    condition = 'Words'
    if VERSION == 'original':
        test_name = f'Control2_{PARTICIPANT_GROUP}_participants_STG_{condition}'
    else:  # revision
        test_name = f'Control2_{PARTICIPANT_GROUP}_participants_STG_revision_{condition}'
    
    # For language comparison, we load both Dutch and Chinese stimuli models
    model_dutch = 'Control2_Delta+Theta_STG_sources_normalized_Dutch_stimuli_acoustic+phonemes'
    model_chinese = 'Control2_Delta+Theta_STG_sources_normalized_Chinese_stimuli_acoustic+phonemes'
    
elif ANALYSIS_TYPE == 'condition_comparison':
    if VERSION == 'original':
        test_name = f'Control2_{PARTICIPANT_GROUP}_participants_{STIMULI_LANGUAGE}_stimuli_Words_vs_syllables'
    else:  # revision
        test_name = f'Control2_{PARTICIPANT_GROUP}_participants_{STIMULI_LANGUAGE}_stimuli_Words_vs_syllables_revision'
    
    # For condition comparison, we use one stimuli language
    model = f'Control2_Delta+Theta_STG_sources_normalized_{STIMULI_LANGUAGE}_stimuli_acoustic+phonemes'
    
else:
    raise ValueError("ANALYSIS_TYPE must be 'language_comparison' or 'condition_comparison'")

# TRF configuration
TRF_nos = [1, 2, 3, 4]
TRF_names = ['spectrogram', 'acoustic_edge', 'phoneme_onset', 'surprisal', 'entropy']

# Statistical parameters
pthresh = 0.05
n_permutations = 30000

print("="*80)
print("TRF ANALYSIS CONFIGURATION")
print("="*80)
print(f"Participant Group: {PARTICIPANT_GROUP} ({len(SUBJECTS)} subjects)")
print(f"Analysis Type: {ANALYSIS_TYPE}")
print(f"Version: {VERSION}")
if ANALYSIS_TYPE == 'condition_comparison':
    print(f"Stimuli Language: {STIMULI_LANGUAGE}")
print(f"Test Name: {test_name}")
print(f"Output Folder: {result_folder}")
print("="*80)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_trf_weights(subject, model_name, condition, trf_index):
    """Load and process TRF weights for a subject."""
    TRF_DIR = subjects_dir / subject / 'meg' / 'TRF' / condition
    trf_lh = eelbrain.load.unpickle(TRF_DIR / f'{subject} {model_name}_lh.pickle')
    trf_rh = eelbrain.load.unpickle(TRF_DIR / f'{subject} {model_name}_rh.pickle')
    
    trf_lh.proportion_explained.source._subjects_dir = subjects_dir
    trf_rh.proportion_explained.source._subjects_dir = subjects_dir
    
    # Process based on TRF index
    if trf_index < 2:
        # For spectrogram and acoustic_edge: average over frequency
        lh_data = np.array([trf_lh.h[trf_index].mean('frequency').square().sqrt().mean('source').x]).T
        rh_data = np.array([trf_rh.h[trf_index].mean('frequency').square().sqrt().mean('source').x]).T
    else:
        # For phoneme_onset, surprisal, entropy: no frequency dimension
        lh_data = np.array([trf_lh.h[trf_index].square().sqrt().mean('source').x]).T
        rh_data = np.array([trf_rh.h[trf_index].square().sqrt().mean('source').x]).T
    
    # Concatenate hemispheres and average
    combined = np.concatenate((lh_data, rh_data), 1).mean(1)
    
    return combined

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\nLoading data...")

if ANALYSIS_TYPE == 'language_comparison':
    # Load data for language comparison (Dutch stimuli vs Chinese stimuli)
    All_Y_Dutch = []
    All_Y_Chinese = []
    
    for i in TRF_nos:
        print(f"  Loading {TRF_names[i]}...")
        
        # Load Dutch stimuli
        rows_dutch = []
        for subject in SUBJECTS:
            weights = load_trf_weights(subject, model_dutch, 'Words', i)
            rows_dutch.append(weights)
        rows_dutch = np.array(rows_dutch)[:, 2:-2]  # Trim edges
        All_Y_Dutch.append(rows_dutch)
        
        # Load Chinese stimuli
        rows_chinese = []
        for subject in SUBJECTS:
            weights = load_trf_weights(subject, model_chinese, 'Words', i)
            rows_chinese.append(weights)
        rows_chinese = np.array(rows_chinese)[:, 2:-2]  # Trim edges
        All_Y_Chinese.append(rows_chinese)
    
    print("Data loaded successfully.\n")
    
    # Determine which tests to run based on version
    if VERSION == 'original':
        print("Running ORIGINAL version tests:")
        print("  Test 1: Acoustic edge (Dutch vs Chinese stimuli)")
        print("  Test 2: Average linguistic features (phoneme_onset + surprisal + entropy)/3")
        
        # Test 1: Acoustic edge
        Y = [All_Y_Dutch[0], All_Y_Chinese[0]]
        print("\nTest 1: Clustering acoustic edge...")
        T_obs, clusters, cluster_p_values, H0 = clu = permutation_cluster_test(
            Y, adjacency=None, n_jobs=-1, threshold=pthresh, max_step=0,
            n_permutations=n_permutations, out_type='indices'
        )
        
        output_file = os.path.join(result_folder, f'clu_{TRF_names[1]}_STG_normalized_{test_name}_whole_brain.pickle')
        with open(output_file, 'wb') as f:
            pickle.dump(clu, f)
        print(f"  Saved: {output_file}")
        
        # Test 2: Average linguistic features
        Y = [(All_Y_Dutch[1] + All_Y_Dutch[2] + All_Y_Dutch[3]) / 3,
             (All_Y_Chinese[1] + All_Y_Chinese[2] + All_Y_Chinese[3]) / 3]
        print("\nTest 2: Clustering average linguistic features...")
        T_obs, clusters, cluster_p_values, H0 = clu = permutation_cluster_test(
            Y, adjacency=None, n_jobs=-1, threshold=pthresh, max_step=0,
            n_permutations=n_permutations, out_type='indices'
        )
        
        output_file = os.path.join(result_folder, f'clu_{TRF_names[2]}_STG_normalized_{test_name}_whole_brain.pickle')
        with open(output_file, 'wb') as f:
            pickle.dump(clu, f)
        print(f"  Saved: {output_file}")
        
        tests_to_save = [1, 2]
        
    else:  # response letter
        print("Running REVISION version tests:")
        print("  Test 1: Phoneme onset (Dutch vs Chinese stimuli)")
        print("  Test 2: Average lexical features (surprisal + entropy)/2")
        
        # Test 1: Phoneme onset
        Y = [All_Y_Dutch[1], All_Y_Chinese[1]]
        print("\nTest 1: Clustering phoneme onset...")
        T_obs, clusters, cluster_p_values, H0 = clu = permutation_cluster_test(
            Y, adjacency=None, n_jobs=-1, threshold=pthresh, max_step=0,
            n_permutations=n_permutations, out_type='indices'
        )
        
        output_file = os.path.join(result_folder, f'clu_{TRF_names[2]}_STG_normalized_{test_name}_whole_brain.pickle')
        with open(output_file, 'wb') as f:
            pickle.dump(clu, f)
        print(f"  Saved: {output_file}")
        
        # Test 2: Average lexical features
        Y = [(All_Y_Dutch[2] + All_Y_Dutch[3]) / 2,
             (All_Y_Chinese[2] + All_Y_Chinese[3]) / 2]
        print("\nTest 2: Clustering average lexical features...")
        T_obs, clusters, cluster_p_values, H0 = clu = permutation_cluster_test(
            Y, adjacency=None, n_jobs=-1, threshold=pthresh, max_step=0,
            n_permutations=n_permutations, out_type='indices'
        )
        
        output_file = os.path.join(result_folder, f'clu_{TRF_names[3]}_STG_normalized_{test_name}_whole_brain.pickle')
        with open(output_file, 'wb') as f:
            pickle.dump(clu, f)
        print(f"  Saved: {output_file}")
        
        tests_to_save = [2, 3]

elif ANALYSIS_TYPE == 'condition_comparison':
    # Load data for condition comparison (Words vs Syllables)
    All_Y_Words = []
    All_Y_Syllables = []
    
    for i in TRF_nos:
        print(f"  Loading {TRF_names[i]}...")
        
        # Load Words condition
        rows_words = []
        for subject in SUBJECTS:
            weights = load_trf_weights(subject, model, 'Words', i)
            rows_words.append(weights)
        rows_words = np.array(rows_words)[:, 2:-2]
        All_Y_Words.append(rows_words)
        
        # Load Syllables condition
        rows_syllables = []
        for subject in SUBJECTS:
            weights = load_trf_weights(subject, model, 'Syllables', i)
            rows_syllables.append(weights)
        rows_syllables = np.array(rows_syllables)[:, 2:-2]
        All_Y_Syllables.append(rows_syllables)
    
    print("Data loaded successfully.\n")
    
    # Determine which tests to run based on version
    if VERSION == 'original':
        print("Running ORIGINAL version tests:")
        print("  Test 1: Acoustic edge (Words vs Syllables)")
        print("  Test 2: Average linguistic features (phoneme_onset + surprisal + entropy)/3")
        
        # Test 1: Acoustic edge
        Y = [All_Y_Words[0], All_Y_Syllables[0]]
        print("\nTest 1: Clustering acoustic edge...")
        T_obs, clusters, cluster_p_values, H0 = clu = permutation_cluster_test(
            Y, adjacency=None, n_jobs=-1, threshold=pthresh,
            n_permutations=n_permutations, out_type='indices'
        )
        
        output_file = os.path.join(result_folder, f'clu_{TRF_names[1]}_STG_normalized_{test_name}_whole_brain.pickle')
        with open(output_file, 'wb') as f:
            pickle.dump(clu, f)
        print(f"  Saved: {output_file}")
        
        # Test 2: Average linguistic features
        Y = [(All_Y_Words[1] + All_Y_Words[2] + All_Y_Words[3]) / 3,
             (All_Y_Syllables[1] + All_Y_Syllables[2] + All_Y_Syllables[3]) / 3]
        print("\nTest 2: Clustering average linguistic features...")
        T_obs, clusters, cluster_p_values, H0 = clu = permutation_cluster_test(
            Y, adjacency=None, n_jobs=-1, threshold=pthresh,
            n_permutations=n_permutations, out_type='indices'
        )
        
        output_file = os.path.join(result_folder, f'clu_{TRF_names[2]}_STG_normalized_{test_name}_whole_brain.pickle')
        with open(output_file, 'wb') as f:
            pickle.dump(clu, f)
        print(f"  Saved: {output_file}")
        
        # All configurations now save both tests
        tests_to_save = [1, 2]
        
    else:  # response letter
        print("Running response letter version tests:")
        print("  Test 1: Phoneme onset (Words vs Syllables)")
        print("  Test 2: Average lexical features (surprisal + entropy)/2")
        
        # Test 1: Phoneme onset
        Y = [All_Y_Words[1], All_Y_Syllables[1]]
        print("\nTest 1: Clustering phoneme onset...")
        T_obs, clusters, cluster_p_values, H0 = clu = permutation_cluster_test(
            Y, adjacency=None, n_jobs=-1, threshold=pthresh,
            n_permutations=n_permutations, out_type='indices'
        )
        
        output_file = os.path.join(result_folder, f'clu_{TRF_names[2]}_STG_normalized_{test_name}_whole_brain.pickle')
        with open(output_file, 'wb') as f:
            pickle.dump(clu, f)
        print(f"  Saved: {output_file}")
        
        # Test 2: Average lexical features
        Y = [(All_Y_Words[2] + All_Y_Words[3]) / 2,
             (All_Y_Syllables[2] + All_Y_Syllables[3]) / 2]
        print("\nTest 2: Clustering average lexical features...")
        T_obs, clusters, cluster_p_values, H0 = clu = permutation_cluster_test(
            Y, adjacency=None, n_jobs=-1, threshold=pthresh,
            n_permutations=n_permutations, out_type='indices'
        )
        
        output_file = os.path.join(result_folder, f'clu_{TRF_names[3]}_STG_normalized_{test_name}_whole_brain.pickle')
        with open(output_file, 'wb') as f:
            pickle.dump(clu, f)
        print(f"  Saved: {output_file}")
        
        tests_to_save = [2, 3]

# ============================================================================
# SAVE SIGNIFICANT TIME POINTS
# ============================================================================

print("\nSaving significant time points...")

n_times = np.arange(-0.05, 0.7, 0.01)

for t in tests_to_save:
    file_name = os.path.join(
        result_folder,
        f'clu_{TRF_names[t]}_STG_normalized_{test_name}_whole_brain.pickle'
    )
    
    with open(file_name, 'rb') as f:
        clu = pickle.load(f)
    
    T_obs, clusters, cluster_p_values, H0 = clu
    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    
    if VERSION == 'revision':
        save_name = f'{TRF_names[t]}_{test_name}_source_whole_brain_revision.npy'
    else:
        save_name = f'{TRF_names[t]}_{test_name}_source_whole_brain.npy'
    
    save_array = np.ones(len(n_times)) * np.NaN
    
    for i in good_cluster_inds:
        for tt in clusters[i]:
            save_array[tt] = 1
    
    output_path = os.path.join(result_folder, save_name)
    np.save(output_path, save_array, allow_pickle=True)
    print(f"  Saved: {output_path}")
    print(f"    Significant clusters: {len(good_cluster_inds)}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
