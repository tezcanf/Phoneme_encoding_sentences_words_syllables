#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:58:38 2021
@author: filtsem

This script compares TRF weights between Sentences and Word_list conditions
using cluster-based permutation tests. It computes the RMS of TRF weights
across hemispheres, then tests for significant temporal clusters where
conditions differ.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import eelbrain
from mne.stats import permutation_cluster_1samp_test


# =============================================================================
# CONFIGURATION
# =============================================================================


# Paths
DATA_ROOT = Path.cwd().parents[1]
SUBJECTS_DIR = DATA_ROOT / 'processed'
RESULT_FOLDER = DATA_ROOT / 'Scripts' / 'TRF_weight_analysis' / 'Output'


MODEL = 'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes+words'
TEST_NAME = 'Control2_Sentence_vs_words'

SUBJECTS = [
    'sub-002', 'sub-003', 'sub-004', 'sub-005', 'sub-006',
    'sub-007', 'sub-008', 'sub-009', 'sub-010', 'sub-011',
    'sub-012', 'sub-013', 'sub-014', 'sub-016', 'sub-017',
    'sub-018', 'sub-019', 'sub-020', 'sub-021'
]

# TRF indices and names (index 0 = spectrogram is skipped)
TRF_INDICES = [1, 2, 3, 4]
TRF_NAMES = ['spectrogram', 'acoustic_edge', 'phoneme_onset', 'surprisal', 'entropy']

# Cluster permutation test parameters
P_THRESHOLD = 0.05
N_PERMUTATIONS = 50000

# Time axis for results
TIME_AXIS = np.arange(-0.05, 0.7, 0.01)


# =============================================================================
# FUNCTIONS
# =============================================================================

def extract_trf_rms(subject, condition, trf_index):
    """
    Extract root-mean-square TRF weights averaged across sources and hemispheres.
    
    For spectral features (index < 2): averages over frequency before RMS.
    For scalar features (index >= 2): computes RMS directly.
    
    Returns a 1D array of RMS values over time.
    """
    trf_dir = SUBJECTS_DIR / subject / 'meg' / 'TRF' / condition
    trf_lh = eelbrain.load.unpickle(trf_dir / f'{subject} {MODEL}_lh.pickle')
    trf_rh = eelbrain.load.unpickle(trf_dir / f'{subject} {MODEL}_rh.pickle')

    # Fix subjects_dir reference
    trf_lh.proportion_explained.source._subjects_dir = SUBJECTS_DIR
    trf_rh.proportion_explained.source._subjects_dir = SUBJECTS_DIR

    if trf_index < 2:
        lh_rms = trf_lh.h[trf_index].mean('frequency').square().sqrt().mean('source').x
        rh_rms = trf_rh.h[trf_index].mean('frequency').square().sqrt().mean('source').x
    else:
        lh_rms = trf_lh.h[trf_index].square().sqrt().mean('source').x
        rh_rms = trf_rh.h[trf_index].square().sqrt().mean('source').x

    # Average across hemispheres: shape (n_times,)
    return np.concatenate(
        (np.array([lh_rms]).T, np.array([rh_rms]).T), axis=1
    ).mean(axis=1)


def collect_trf_weights(condition):
    """
    Collect TRF RMS weights for all subjects and TRF features for a given condition.
    
    Returns a list of arrays, one per TRF feature, each with shape (n_subjects, n_times).
    Time points are trimmed by 2 samples on each end.
    """
    all_weights = []

    for trf_idx in TRF_INDICES:
        print(f"  Extracting {TRF_NAMES[trf_idx]} for {condition}...")
        subject_data = []
        for subject in SUBJECTS:
            rms = extract_trf_rms(subject, condition, trf_idx)
            subject_data.append(rms)

        # Trim 2 time points from each end
        weights = np.array(subject_data)[:, 2:-2]
        all_weights.append(weights)

    return all_weights


def run_cluster_test(Y, trf_name):
    """
    Run a 1-sample cluster permutation test on the difference array Y
    and save results to a pickle file.
    """
    print(f"  Running cluster permutation test for {trf_name}...")
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        Y, adjacency=None, n_jobs=-1,
        threshold=P_THRESHOLD,
        n_permutations=N_PERMUTATIONS,
        out_type='indices'
    )
    clu = (T_obs, clusters, cluster_p_values, H0)

    out_path = os.path.join(
        RESULT_FOLDER, f'clu_{trf_name}_STG_normalized_{TEST_NAME}.pickle'
    )
    with open(out_path, 'wb') as f:
        pickle.dump(clu, f)
    print(f"  Saved: {out_path}")

    return clu


def extract_significant_clusters(trf_name):
    """
    Load cluster test results and create a binary mask of significant time points.
    Saves the mask as a .npy file.
    """
    file_path = os.path.join(
        RESULT_FOLDER, f'clu_{trf_name}_STG_normalized_{TEST_NAME}.pickle'
    )
    with open(file_path, 'rb') as f:
        T_obs, clusters, cluster_p_values, H0 = pickle.load(f)

    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

    sig_mask = np.ones(len(TIME_AXIS)) * np.nan
    for i in good_cluster_inds:
        for tt in clusters[i]:
            sig_mask[tt] = 1

    save_path = os.path.join(
        RESULT_FOLDER, f'{trf_name}_{TEST_NAME}_source_STG.npy'
    )
    np.save(save_path, sig_mask, allow_pickle=True)
    print(f"  Significant cluster mask saved: {save_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Step 1: Collect TRF weights for both conditions
    print("=" * 60)
    print("Collecting TRF weights for Sentences condition...")
    weights_sentences = collect_trf_weights('Sentences')

    print("Collecting TRF weights for Word_list condition...")
    weights_wordlist = collect_trf_weights('Word_list')

    # Step 2: Run cluster permutation tests
    # TRF_INDICES maps to: [1]=acoustic_edge, [2]=phoneme_onset, [3]=surprisal, [4]=entropy
    # weights lists are indexed 0-3 corresponding to TRF_INDICES 1-4

    print("=" * 60)
    print("Running cluster permutation tests...")

    # Test 1: Acoustic edge (index 0 in weights = TRF index 1)
    Y_acoustic = weights_sentences[0] - weights_wordlist[0]
    run_cluster_test(Y_acoustic, TRF_NAMES[1])

    # Test 2: Average of phoneme_onset, surprisal, entropy (indices 1-3 in weights)
    Y_phoneme_avg = (
        (weights_sentences[1] + weights_sentences[2] + weights_sentences[3]) / 3
        - (weights_wordlist[1] + weights_wordlist[2] + weights_wordlist[3]) / 3
    )
    run_cluster_test(Y_phoneme_avg, TRF_NAMES[2])

    # Step 3: Extract significant clusters
    print("=" * 60)
    print("Extracting significant clusters...")
    for trf_name in [TRF_NAMES[1], TRF_NAMES[2]]:
        extract_significant_clusters(trf_name)

    print("=" * 60)
    print("ANALYSIS COMPLETE")