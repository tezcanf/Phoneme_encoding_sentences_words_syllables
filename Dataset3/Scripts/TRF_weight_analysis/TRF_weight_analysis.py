#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster-based permutation testing for TRF comparisons across conditions.
Performs statistical testing on TRF weights averaged across STG sources.

Supports two types of comparisons:
1. Sentences vs Words
2. Words vs Syllables

@author: filiztezcan
"""

from pathlib import Path
import numpy as np
import eelbrain
import pickle
from mne.stats import permutation_cluster_1samp_test

# =============================================================================
# Configuration
# =============================================================================

# Paths
root = Path.cwd().parents[1]
SUBJECTS_DIR = root  / 'processed' 
RESULT_FOLDER = root / 'Scripts' / 'TRF_weight_analysis' / 'Output'
DATA_ROOT = root / 'TRF_models'


SUBJECTS = [
    'sub-006', 'sub-008', 'sub-009', 'sub-010', 'sub-011', 'sub-012',
    'sub-013', 'sub-015', 'sub-016', 'sub-017', 'sub-018', 'sub-019',
    'sub-020', 'sub-021', 'sub-022', 'sub-023', 'sub-024', 'sub-025',
    'sub-026', 'sub-027', 'sub-028', 'sub-030', 'sub-031', 'sub-032',
    'sub-033', 'sub-034', 'sub-035', 'sub-036', 'sub-037', 'sub-038'
]

TRF_INDICES = [1, 2, 3, 4]  # Skip index 0 (spectrogram)
TRF_NAMES = ['spectrogram', 'acoustic_edge', 'phoneme_onset', 'surprisal', 'entropy']

# Statistical parameters
P_THRESHOLD = 0.05
N_PERMUTATIONS = 30000

# Time vector for results
N_TIMES = np.arange(-0.05, 0.7, 0.01)

# Comparison configurations
COMPARISONS = {
    'sentences_vs_words': {
        'condition1': 'control_sentence',
        'condition2': 'random_word_list',
        'model': 'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes+words',
        'test_name': 'Control2_Sentence_vs_words'
    },
    'words_vs_syllables': {
        'condition1': 'control_sentence',  # Actually words in this context
        'condition2': 'random_word_list',   # Actually syllables in this context
        'model': 'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes',
        'test_name': 'Control2_Words_vs_syllables'
    }
}

# =============================================================================
# Helper Functions
# =============================================================================

def load_and_process_trf(subject, condition, model, trf_index):
    """
    Load TRF data for a subject and extract averaged weights.
    
    Parameters
    ----------
    subject : str
        Subject ID
    condition : str
        Condition name (e.g., 'control_sentence', 'random_word_list')
    model : str
        Model name string
    trf_index : int
        Index of TRF component to extract
    
    Returns
    -------
    ndarray
        Time-averaged TRF weights across sources and hemispheres
    """
    trf_dir = DATA_ROOT / condition / subject
    
    # Load left and right hemisphere TRFs
    trf_lh = eelbrain.load.unpickle(trf_dir / f'{subject} {model}_lh.pickle')
    trf_rh = eelbrain.load.unpickle(trf_dir / f'{subject} {model}_rh.pickle')
    
    # Set subjects directory
    trf_lh.proportion_explained.source._subjects_dir = SUBJECTS_DIR
    trf_rh.proportion_explained.source._subjects_dir = SUBJECTS_DIR
    
    # Extract and process weights based on TRF type
    if trf_index < 2:  # Frequency-based TRFs (spectrogram, acoustic_edge)
        weights_lh = trf_lh.h[trf_index].mean('frequency').square().sqrt().mean('source').x
        weights_rh = trf_rh.h[trf_index].mean('frequency').square().sqrt().mean('source').x
    else:  # Non-frequency TRFs (phoneme_onset, surprisal, entropy)
        weights_lh = trf_lh.h[trf_index].square().sqrt().mean('source').x
        weights_rh = trf_rh.h[trf_index].square().sqrt().mean('source').x
    
    # Average across hemispheres
    combined_weights = np.concatenate(
        (np.array([weights_lh]).T, np.array([weights_rh]).T), axis=1
    ).mean(axis=1)
    
    return combined_weights


def load_condition_data(condition, model, trf_indices):
    """
    Load TRF data for all subjects in a condition.
    
    Parameters
    ----------
    condition : str
        Condition name
    model : str
        Model name string
    trf_indices : list
        List of TRF indices to load
    
    Returns
    -------
    list of ndarray
        TRF data for each component, shape (n_subjects, n_times)
    """
    all_data = []
    
    for trf_idx in trf_indices:
        print(f'  Loading {TRF_NAMES[trf_idx]}...')
        subject_data = []
        
        for subject in SUBJECTS:
            weights = load_and_process_trf(subject, condition, model, trf_idx)
            subject_data.append(weights)
        
        # Stack subjects and trim first 2 and last 2 time points
        subject_data = np.array(subject_data)[:, 2:-2]
        all_data.append(subject_data)
    
    return all_data


def run_cluster_permutation_test(data, output_filename):
    """
    Run cluster-based permutation test and save results.
    
    Parameters
    ----------
    data : ndarray
        Data to test, shape (n_subjects, n_times)
    output_filename : str
        Full filename for the output pickle file
    
    Returns
    -------
    tuple
        (T_obs, clusters, cluster_p_values, H0)
    """
    print('  Running cluster permutation test...')
    
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        data,
        adjacency=None,
        n_jobs=-1,
        threshold=P_THRESHOLD,
        n_permutations=N_PERMUTATIONS,
        out_type='indices'
    )
    
    # Save cluster results
    with open(RESULT_FOLDER / output_filename, 'wb') as f:
        pickle.dump((T_obs, clusters, cluster_p_values, H0), f)
    
    print(f'  Saved: {output_filename}')
    return T_obs, clusters, cluster_p_values, H0


def extract_and_save_significant_timepoints(input_filename, output_filename):
    """
    Extract significant time points from cluster results and save as binary mask.
    
    Parameters
    ----------
    input_filename : str
        Name of the input pickle file with cluster results
    output_filename : str
        Name of the output numpy file for the significance mask
    """
    with open(RESULT_FOLDER / input_filename, 'rb') as f:
        T_obs, clusters, cluster_p_values, H0 = pickle.load(f)
    
    # Find significant clusters
    significant_clusters = np.where(cluster_p_values < 0.05)[0]
    
    # Create binary mask
    sig_mask = np.ones(len(N_TIMES)) * np.nan
    
    for cluster_idx in significant_clusters:
        for time_idx in clusters[cluster_idx]:
            sig_mask[time_idx] = 1
    
    # Save mask
    np.save(RESULT_FOLDER / output_filename, sig_mask, allow_pickle=True)
    
    print(f'  Saved significant timepoints: {output_filename}')


def run_comparison(comparison_key):
    """
    Run complete analysis pipeline for a specific comparison.
    
    Parameters
    ----------
    comparison_key : str
        Key from COMPARISONS dict ('sentences_vs_words' or 'words_vs_syllables')
    """
    config = COMPARISONS[comparison_key]
    print(f"\n{'='*70}")
    print(f"Running analysis: {config['test_name']}")
    print(f"{'='*70}")
    
    # Load data for both conditions
    print(f"\nLoading {config['condition1']} data...")
    condition1_data = load_condition_data(
        config['condition1'], 
        config['model'], 
        TRF_INDICES
    )
    
    print(f"\nLoading {config['condition2']} data...")
    condition2_data = load_condition_data(
        config['condition2'], 
        config['model'], 
        TRF_INDICES
    )
    
    # Test 1: Acoustic edge (index 1 in TRF_NAMES)
    print(f"\n--- Testing {TRF_NAMES[1]} ---")
    contrast_acoustic = condition1_data[0] - condition2_data[0]
    output_file = f"clu_{TRF_NAMES[1]}_STG_normalized_{config['test_name']}_whole_brain.pickle"
    run_cluster_permutation_test(contrast_acoustic, output_file)
    
    # Test 2: Average of phoneme_onset, surprisal, and entropy
    print(f"\n--- Testing average of {TRF_NAMES[2]}, {TRF_NAMES[3]}, {TRF_NAMES[4]} ---")
    condition1_avg = np.mean(
        [condition1_data[1], condition1_data[2], condition1_data[3]], 
        axis=0
    )
    condition2_avg = np.mean(
        [condition2_data[1], condition2_data[2], condition2_data[3]], 
        axis=0
    )
    contrast_combined = condition1_avg - condition2_avg
    output_file = f"clu_{TRF_NAMES[2]}_STG_normalized_{config['test_name']}_whole_brain.pickle"
    run_cluster_permutation_test(contrast_combined, output_file)
    
    # Extract and save significant timepoints for both tests
    print('\n--- Extracting significant timepoints ---')
    for trf_name in [TRF_NAMES[1], TRF_NAMES[2]]:
        input_file = f"clu_{trf_name}_STG_normalized_{config['test_name']}_whole_brain.pickle"
        output_file = f"{trf_name}_{config['test_name']}_source_whole_brain.npy"
        extract_and_save_significant_timepoints(input_file, output_file)
    
    print(f"\nCompleted: {config['test_name']}")


# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == '__main__':
    
    # Create result folder if it doesn't exist
    RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Run both comparisons
    for comparison_key in COMPARISONS.keys():
        run_comparison(comparison_key)
    
    print("\n" + "="*70)
    print("All analyses complete!")
    print("="*70)