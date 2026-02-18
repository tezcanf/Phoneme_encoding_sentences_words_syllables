#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRF Condition Comparison Analysis

Compares TRF weights across different linguistic conditions (sentences vs words vs syllables)
for acoustic and phoneme features.

@author: filiztezcan
"""

from pathlib import Path
import numpy as np
import eelbrain
import matplotlib.pyplot as plt
from scipy.stats import sem
from matplotlib.ticker import MaxNLocator


# ============================================================================
# Configuration
# ============================================================================

# Paths
root = Path.cwd().parents[1]
SUBJECTS_DIR = root  / 'processed' 
ANOVA_RESULTS_PATH = root / 'Scripts' / 'TRF_weight_analysis' / 'Output'
DATA_ROOT = root / 'TRF_models'
RESULTS_PATH = root / 'Scripts' / 'Visualize_TRF_weights' / 'Output'

# Subject list
SUBJECTS = [
    'sub-006', 'sub-008', 'sub-009', 'sub-010', 'sub-011', 'sub-012', 'sub-013',
    'sub-015', 'sub-016', 'sub-017', 'sub-018', 'sub-019', 'sub-020', 'sub-021',
    'sub-022', 'sub-023', 'sub-024', 'sub-025', 'sub-026', 'sub-027', 'sub-028',
    'sub-030', 'sub-031', 'sub-032', 'sub-033', 'sub-034', 'sub-035', 'sub-036',
    'sub-037', 'sub-038'
]

# Plotting configuration
FONT = "Times New Roman"
RC = {
    'figure.dpi': 100,
    'savefig.dpi': 100,
    'savefig.transparent': True,
    'font.family': FONT,
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 10,
    'figure.figsize': (2, 1.5)
}
plt.rcParams.update(RC)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_condition_data(condition, model_name, hemisphere):
    """
    Load TRF data for a specific condition and hemisphere.
    
    Parameters
    ----------
    condition : str
        Condition name (e.g., 'control_sentence', 'random_word_list', 'random_syllables')
    model_name : str
        Model specification string
    hemisphere : str
        'lh' or 'rh'
    
    Returns
    -------
    eelbrain.Dataset
        Dataset containing subject data and TRF weights
    """
    rows = []
    x_names = None
    
    for subject in SUBJECTS:
        print(f"Loading {subject} - {condition} - {hemisphere}")
        trf_dir = DATA_ROOT / condition / subject
        trf = eelbrain.load.unpickle(trf_dir / f'{subject} {model_name}_{hemisphere}.pickle')
        trf.r.source._subjects_dir = SUBJECTS_DIR
        rows.append([subject, trf.r, *trf.h])
        x_names = trf.x
    
    return eelbrain.Dataset.from_caselist(['subject', 'det', *x_names], rows)


def load_bilateral_data(condition, model_name):
    """
    Load TRF data for both hemispheres.
    
    Parameters
    ----------
    condition : str
        Condition name
    model_name : str
        Model specification string
    
    Returns
    -------
    tuple
        (left_hemisphere_data, right_hemisphere_data)
    """
    data_lh = load_condition_data(condition, model_name, 'lh')
    data_rh = load_condition_data(condition, model_name, 'rh')
    return data_lh, data_rh


# ============================================================================
# Feature Extraction Functions
# ============================================================================

def compute_acoustic_feature(data_lh, data_rh):
    """
    Compute acoustic edge features (averaged across hemispheres and frequency).
    
    Parameters
    ----------
    data_lh, data_rh : eelbrain.Dataset
        Left and right hemisphere data
    
    Returns
    -------
    eelbrain array
        Averaged acoustic feature
    """
    lh_acoustic = data_lh["gammatone_on"].square().sqrt().mean('frequency').mean('source')
    rh_acoustic = data_rh["gammatone_on"].square().sqrt().mean('frequency').mean('source')
    return (lh_acoustic + rh_acoustic) / 2


def compute_phoneme_features(data_lh, data_rh):
    """
    Compute phoneme-related features (averaged across hemispheres).
    
    Includes phonemes, cohort surprisal, and cohort entropy.
    
    Parameters
    ----------
    data_lh, data_rh : eelbrain.Dataset
        Left and right hemisphere data
    
    Returns
    -------
    eelbrain array
        Averaged phoneme feature
    """
    lh_features = (
        data_lh["phonemes"].square().sqrt().mean('source') +
        data_lh["cohort_surprisal"].square().sqrt().mean('source') +
        data_lh["cohort_entropy"].square().sqrt().mean('source')
    )
    
    rh_features = (
        data_rh["phonemes"].square().sqrt().mean('source') +
        data_rh["cohort_surprisal"].square().sqrt().mean('source') +
        data_rh["cohort_entropy"].square().sqrt().mean('source')
    )
    
    return (lh_features + rh_features) / 6


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_condition_comparison(x1_all, x2_all, titles, condition_p_all, 
                              color1, color2, output_path=None):
    """
    Create comparison plots for multiple features.
    
    Parameters
    ----------
    x1_all, x2_all : list
        Lists of data arrays for each condition
    titles : list
        List of subplot titles
    condition_p_all : list
        List of significance arrays
    color1, color2 : str
        Hex colors for the two conditions
    output_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(2.8, 3), sharex=True, sharey=True)
    
    for i, ax in enumerate(axes):
        x1 = x1_all[i]
        x2 = x2_all[i]
        condition_p = condition_p_all[i]
        
        # Prepare data (trim edges)
        x1_mean = x1.mean('case').x[2:-2]
        x2_mean = x2.mean('case').x[2:-2]
        time = x1.time.times[2:-2]
        
        # Compute standard errors
        x1_error = [sem(x1.x[:, t]) for t in range(len(time))]
        x2_error = [sem(x2.x[:, t]) for t in range(len(time))]
        
        # Plot condition 1
        ax.plot(time, x1_mean, color1, linewidth=0.5)
        ax.fill_between(time, x1_mean - x1_error, x1_mean + x1_error,
                        alpha=0.7, color=color1, linewidth=0.0)
        
        # Plot condition 2
        ax.plot(time, x2_mean, color2, linewidth=0.5)
        ax.fill_between(time, x2_mean - x2_error, x2_mean + x2_error,
                        alpha=0.7, color=color2, linewidth=0.0)
        
        # Plot significance markers
        ax.plot(time, np.multiply(condition_p, -0.00), 'red', linewidth=2)
        
        # Format axes
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.grid()
        ax.xaxis.grid()
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_title(titles[i])
    
    plt.subplots_adjust(hspace=0.0)
    fig.text(0.5, 0., 'Time (sec)', ha='center', fontsize=12)
    fig.text(0, 0.5, 'Power of Weights $\mathregular{\sqrt{w^{2}}}$',
             va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path,format='svg')
    plt.show()


# ============================================================================
# Main Analysis Functions
# ============================================================================

def compare_sentences_vs_words():
    """
    Compare sentence and word conditions.
    """
    print("\n" + "="*60)
    print("ANALYSIS: Sentences vs Words")
    print("="*60 + "\n")
    
    model_name = 'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes+words'
    test_name = 'Control2_Sentence_vs_words'
    
    # Load data
    print("Loading sentence data...")
    sentence_lh, sentence_rh = load_bilateral_data('control_sentence', model_name)
    
    print("Loading word data...")
    word_lh, word_rh = load_bilateral_data('random_word_list', model_name)
    
    # Extract features
    print("Computing features...")
    sentence_acoustic = compute_acoustic_feature(sentence_lh, sentence_rh)
    word_acoustic = compute_acoustic_feature(word_lh, word_rh)
    
    sentence_phoneme = compute_phoneme_features(sentence_lh, sentence_rh)
    word_phoneme = compute_phoneme_features(word_lh, word_rh)
    
    # Load significance results
    acoustic_p = np.load(
        f'{ANOVA_RESULTS_PATH}/acoustic_edge_{test_name}_source_whole_brain.npy',
        allow_pickle=True
    )
    phoneme_p = np.load(
        f'{ANOVA_RESULTS_PATH}/phoneme_onset_{test_name}_source_whole_brain.npy',
        allow_pickle=True
    )
    
    # Plot results
    print("Generating plots...")
    output_file = RESULTS_PATH / f'{test_name}_comparison.svg'
    plot_condition_comparison(
        x1_all=[sentence_acoustic, sentence_phoneme],
        x2_all=[word_acoustic, word_phoneme],
        titles=['Acoustic Edges', 'Phoneme Features'],
        condition_p_all=[acoustic_p, phoneme_p],
        color1='#d9a359',  # Sentences
        color2='#2f4858',   # Words
        output_path=output_file
    )
    print(f"Figure saved to: {output_file}")


def compare_words_vs_syllables():
    """
    Compare word and syllable conditions.
    """
    print("\n" + "="*60)
    print("ANALYSIS: Words vs Syllables")
    print("="*60 + "\n")
    
    model_name = 'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes'
    test_name = 'Control2_Words_vs_syllables'
    
    # Load data
    print("Loading word data...")
    word_lh, word_rh = load_bilateral_data('random_word_list', model_name)
    
    print("Loading syllable data...")
    syllable_lh, syllable_rh = load_bilateral_data('random_syllables', model_name)
    
    # Extract features
    print("Computing features...")
    word_acoustic = compute_acoustic_feature(word_lh, word_rh)
    syllable_acoustic = compute_acoustic_feature(syllable_lh, syllable_rh)
    
    word_phoneme = compute_phoneme_features(word_lh, word_rh)
    syllable_phoneme = compute_phoneme_features(syllable_lh, syllable_rh)
    
    # Load significance results
    acoustic_p = np.load(
        f'{ANOVA_RESULTS_PATH}/acoustic_edge_{test_name}_source_whole_brain.npy',
        allow_pickle=True
    )
    phoneme_p = np.load(
        f'{ANOVA_RESULTS_PATH}/phoneme_onset_{test_name}_source_whole_brain.npy',
        allow_pickle=True
    )
    
    # Plot results
    print("Generating plots...")
    output_file = RESULTS_PATH / f'{test_name}_comparison.svg'
    plot_condition_comparison(
        x1_all=[word_acoustic, word_phoneme],
        x2_all=[syllable_acoustic, syllable_phoneme],
        titles=['Acoustic Edges', 'Phoneme Features'],
        condition_p_all=[acoustic_p, phoneme_p],
        color1='#2f4858',  # Words
        color2='#719e87',   # Syllables
        output_path=output_file
    )
    print(f"Figure saved to: {output_file}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Run both analyses
    compare_sentences_vs_words()
    compare_words_vs_syllables()