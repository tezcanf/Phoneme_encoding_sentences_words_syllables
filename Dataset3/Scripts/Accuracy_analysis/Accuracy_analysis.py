#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRF Accuracy Analysis: Model Comparison
Calculates explained variance (R²) improvements by comparing baseline and full models
across different experimental conditions and linguistic features.

@author: filiztezcan
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.patches import Patch
import eelbrain


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
root = Path.cwd().parents[1]
SUBJECT_DIR = root / 'processed'
RESULTS_PATH = root / 'Scripts' / 'Accuracy_analysis' / 'Results_publication'
DATA_ROOT = root / 'TRF_models'


# Ensure output directory exists
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Subject list
SUBJECTS = [
    'sub-006', 'sub-008', 'sub-009', 'sub-010', 'sub-011',
    'sub-012', 'sub-013', 'sub-015', 'sub-016', 'sub-017',
    'sub-018', 'sub-019', 'sub-020', 'sub-021', 'sub-022',
    'sub-023', 'sub-024', 'sub-025', 'sub-026', 'sub-027',
    'sub-028', 'sub-030', 'sub-031', 'sub-032', 'sub-033',
    'sub-034', 'sub-035', 'sub-036', 'sub-037', 'sub-038'
]

# Analysis configurations
ANALYSES = {
    'words_vs_syllables': {
        'conditions': ['random_word_list', 'random_syllables'],
        'features': ['Acoustic Edge', 'Phoneme Onset'],
        'models': [
            # Acoustic Edge: phonemes → acoustic+phonemes
            ['Control2_Delta+Theta_STG_sources_normalized_phonemes',
             'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes'],
            # Phoneme Onset: acoustic → acoustic+phonemes
            ['Control2_Delta+Theta_STG_sources_normalized_acoustic',
             'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes']
        ],
        'colors': {'random_word_list': '#2f4858', 'random_syllables': '#719e87'},
        'labels': {'random_word_list': 'Word Lists', 'random_syllables': 'Random Syllables'},
        'title': 'Word Lists vs Random Syllables',
        'sig_signs': ['n.s.', '****', '****', '*'],
        'output_csv': 'Dataset3_Accuracies_both_hemispheres_words_vs_syllables_STG_Delta_Theta.csv',
        'output_fig': 'Accuracy_words_vs_syllables.svg'
    },
    'sentences_vs_words': {
        'conditions': ['control_sentence', 'random_word_list'],
        'features': ['Acoustic Edge', 'Phoneme Onset'],
        'models': [
            # Acoustic Edge: phonemes+words → acoustic+phonemes+words
            ['Control2_Delta+Theta_STG_sources_normalized_phonemes+words',
             'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes+words'],
            # Phoneme Onset: acoustic+words → acoustic+phonemes+words
            ['Control2_Delta+Theta_STG_sources_normalized_acoustic+words',
             'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes+words']
        ],
        'colors': {'control_sentence': '#d9a359', 'random_word_list': '#2f4858'},
        'labels': {'control_sentence': 'Sentences', 'random_word_list': 'Word Lists'},
        'title': 'Sentences vs Word Lists',
        'sig_signs': ['.', '.', 'n.s.', 'n.s.'],
        'output_csv': 'Dataset3_Accuracies_both_hemispheres_sentences_vs_words_STG_Delta_Theta.csv',
        'output_fig': 'Accuracy_sentences_vs_words.svg'
    }
}

# Plotting configuration
FONT = "Times New Roman"
RC_PARAMS = {
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.transparent': True,
    'font.family': FONT,
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'figure.figsize': (3, 3),
}
pyplot.rcParams.update(RC_PARAMS)


# ============================================================================
# FUNCTIONS
# ============================================================================

def calculate_accuracy(subject, condition, models):
    """
    Calculate accuracy improvement (R²) by subtracting baseline model
    from full model, averaged across both hemispheres.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-006')
    condition : str
        Experimental condition (e.g., 'random_word_list')
    models : list of str
        [baseline_model, full_model] names
        
    Returns
    -------
    float
        Mean accuracy improvement across hemispheres
    """
    trf_dir = DATA_ROOT / condition / subject
    
    # Load TRF models for both hemispheres
    trf_lh_base = eelbrain.load.unpickle(trf_dir / f'{subject} {models[0]}_lh.pickle')
    trf_rh_base = eelbrain.load.unpickle(trf_dir / f'{subject} {models[0]}_rh.pickle')
    trf_lh_full = eelbrain.load.unpickle(trf_dir / f'{subject} {models[1]}_lh.pickle')
    trf_rh_full = eelbrain.load.unpickle(trf_dir / f'{subject} {models[1]}_rh.pickle')
    
    # Compute difference in proportion explained (R²)
    diff_lh = trf_lh_full.proportion_explained.x - trf_lh_base.proportion_explained.x
    diff_rh = trf_rh_full.proportion_explained.x - trf_rh_base.proportion_explained.x
    
    # Fix subjects_dir reference for source space
    trf_lh_full.proportion_explained.source._subjects_dir = SUBJECT_DIR
    trf_rh_full.proportion_explained.source._subjects_dir = SUBJECT_DIR
    
    return np.mean([np.mean(diff_lh), np.mean(diff_rh)])


def collect_all_data(analysis_config):
    """
    Collect accuracy data for all subjects, conditions, and features.
    
    Parameters
    ----------
    analysis_config : dict
        Configuration dictionary containing conditions, features, and models
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Condition, Feature, subject, accuracy
    """
    rows = []
    features = analysis_config['features']
    conditions = analysis_config['conditions']
    models_list = analysis_config['models']
    
    for feature_idx, feature in enumerate(features):
        models = models_list[feature_idx]
        for condition in conditions:
            for subject in SUBJECTS:
                print(f"Processing {subject} - {condition} - {feature}")
                accuracy = calculate_accuracy(subject, condition, models)
                rows.append({
                    'Condition': condition,
                    'Feature': feature,
                    'subject': subject,
                    'accuracy': accuracy
                })
    
    return pd.DataFrame(rows)


def get_whisker_heights(box_data):
    """
    Calculate the upper whisker height for each box.
    Whisker extends to the highest data point within 1.5*IQR above Q3.
    
    Parameters
    ----------
    box_data : list of arrays
        List of data arrays for each box
        
    Returns
    -------
    list
        Upper whisker heights for positioning significance bars
    """
    heights = []
    for data in box_data:
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        upper_whisker = q3 + 1.5 * iqr
        # Whisker extends to highest data point within range
        whisker_height = np.max(data[data <= upper_whisker])
        heights.append(whisker_height)
    return heights


def add_significance_bars(ax, positions, sig_signs, whisker_heights):
    """
    Add significance bracket indicators to the boxplot with dynamic positioning.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object to add bars to
    positions : list
        X positions of boxes
    sig_signs : list
        Significance labels in order: [comparison1, comparison2, comparison3, comparison4]
        Comparisons: [cond1_feat1 vs cond2_feat1, cond1_feat2 vs cond2_feat2,
                      feat1 vs feat2 for cond1, feat1 vs feat2 for cond2]
    whisker_heights : list
        Upper whisker heights of each box for positioning
    """
    col = 'k'
    
    # Calculate base height and spacing dynamically based on whisker heights
    max_height = max(whisker_heights)
    h = max_height * 0.015  # Bracket height as proportion of max
    y_offset = max_height * 0.05  # Base offset above whiskers
    spacing = max_height * 0.1  # Spacing between bracket levels
    
    # Base positions for different comparison levels
    y_base = max_height + y_offset
    y_positions = [
        y_base,                    # Level 1: within-feature between conditions
        y_base,                    # Level 1: within-feature between conditions
        y_base + spacing,          # Level 2: within-condition between features
        y_base + 1.8 * spacing     # Level 3: within-condition between features
    ]
    
    # Comparisons: (pos1, pos2, sig_index)
    comparisons = [
        (positions[0], positions[1], 0),  # Feature 1: Condition 1 vs 2
        (positions[2], positions[3], 1),  # Feature 2: Condition 1 vs 2
        (positions[0], positions[2], 2),  # Condition 1: Feature 1 vs 2
        (positions[1], positions[3], 3),  # Condition 2: Feature 1 vs 2
    ]
    
    for (x1, x2, sig_idx), y in zip(comparisons, y_positions):
        # Draw bracket
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.5, c=col)
        # Add significance label
        ax.text((x1 + x2) * 0.5, y + h, sig_signs[sig_idx],
                ha='center', va='bottom', color=col, fontsize=8)


def plot_boxplot(df, analysis_config, save_path=None):
    """
    Create and save box plot comparing conditions across features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with columns: Condition, Feature, subject, accuracy
    analysis_config : dict
        Configuration dictionary with plotting parameters
    save_path : str or Path, optional
        Path to save figure. If None, only displays.
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    features = analysis_config['features']
    conditions = analysis_config['conditions']
    colors = analysis_config['colors']
    labels = analysis_config['labels']
    title = analysis_config['title']
    sig_signs = analysis_config['sig_signs']
    
    # Prepare data for boxplot
    box_data = []
    positions = []
    colors_list = []
    pos_counter = 0
    
    for feature in features:
        for condition in conditions:
            data = df[(df.Feature == feature) & 
                     (df.Condition == condition)].accuracy.values
            box_data.append(data)
            positions.append(pos_counter)
            colors_list.append(colors[condition])
            pos_counter += 1
        pos_counter += 0.5  # Gap between feature groups
    
    # Get whisker heights for dynamic positioning
    whisker_heights = get_whisker_heights(box_data)
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Create boxplot
    bp = ax.boxplot(box_data, positions=positions, widths=0.4,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    boxprops=dict(linewidth=0.8),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0))
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add significance bars
    add_significance_bars(ax, positions, sig_signs, whisker_heights)
    
    # Styling
    ax.set_xticks([0.5, 3])
    ax.set_xticklabels(['Acoustic Edge', 'Phonemes'], fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_ylabel('Accuracy improvement $\\mathregular{R^{2}}$', fontsize=12)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    
    # Legend
    legend_elements = [
        Patch(facecolor=colors[cond], edgecolor='black', 
              label=labels[cond], alpha=0.7)
        for cond in conditions
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',format='svg')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    return fig


def run_statistical_tests(df, features, conditions):
    """
    Run one-sample t-tests against zero for each feature × condition.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with columns: Feature, Condition, accuracy
    features : list
        List of feature names
    conditions : list
        List of condition names
    """
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (one-sample t-test against 0)")
    print("=" * 70)
    
    for feature in features:
        for condition in conditions:
            data = df[(df.Feature == feature) & 
                     (df.Condition == condition)]['accuracy'].tolist()
            result = scipy.stats.ttest_1samp(data, 0)
            print(f"\n{feature} - {condition}:")
            print(f"  Mean = {np.mean(data):.6f}")
            print(f"  t({len(data)-1}) = {result.statistic:.4f}, p = {result.pvalue:.4e}")


def run_analysis(analysis_name):
    """
    Run complete analysis pipeline for a given analysis configuration.
    
    Parameters
    ----------
    analysis_name : str
        Name of analysis ('words_vs_syllables' or 'sentences_vs_words')
    """
    config = ANALYSES[analysis_name]
    
    print("\n" + "=" * 70)
    print(f"ANALYSIS: {config['title'].upper()}")
    print("=" * 70)
    
    # Collect data
    print("\n--- Collecting Data ---")
    df = collect_all_data(config)
    
    # Save data
    csv_path = RESULTS_PATH / config['output_csv']
    df.to_csv(csv_path, index=False)
    print(f"\nData saved to: {csv_path}")
    
    # Create and save plot
    print("\n--- Creating Plot ---")
    fig_path = RESULTS_PATH / config['output_fig']
    fig = plot_boxplot(df, config, save_path=fig_path)
    
    # Run statistics
    run_statistical_tests(df, config['features'], config['conditions'])
    
    return df, fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TRF ACCURACY ANALYSIS")
    print("=" * 70)
    
    # Run both analyses
    results = {}
    
    # Analysis 1: Words vs Syllables
    results['words_vs_syllables'] = run_analysis('words_vs_syllables')
    
    # Analysis 2: Sentences vs Words
    results['sentences_vs_words'] = run_analysis('sentences_vs_words')
    
    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)
    print(f"\nResults saved in: {RESULTS_PATH}")
    