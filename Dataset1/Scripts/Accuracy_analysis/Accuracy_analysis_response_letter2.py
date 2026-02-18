#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRF Accuracy Analysis: Shuffled vs Normal Phoneme Features Comparison

This script calculates the explained variance of phoneme features by comparing
normal (incremental) phoneme processing versus shuffled phoneme processing for
both Sentences and Word_list conditions.

Author: filtsem

"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from cycler import cycler
import scipy.stats
from scipy.stats import sem
import eelbrain

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
root = Path.cwd().parents[1]
SUBJECT_DIR = root / 'processed'

# SUBJECT_DIR = Path('/project/3027003.01/Filiz_folders_dont_delete/Sanne_MEG/processed/')
RESULTS_PATH = root / 'Scripts' / 'Accuracy_analysis' / 'Results_publication_revision'

# Subject list
SUBJECTS = [
    'sub-002', 'sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007',
    'sub-008', 'sub-009', 'sub-010', 'sub-011', 'sub-012', 'sub-013',
    'sub-014', 'sub-016', 'sub-017', 'sub-018', 'sub-019', 'sub-020',
    'sub-021'
]



# Choose which analysis to run: 'phoneme_onset' or 'phoneme_surprisal_entropy'
ANALYSIS_TYPE = 'phoneme_surprisal_entropy'  # Change to 'phoneme_onset' for onset analysis

# Model comparisons: [baseline_model, full_model]
# Feature contribution = full_model - baseline_model

MODEL_COMPARISONS = {
    'Phoneme Surp./Ent.': ['Control2_Delta+Theta_STG_sources_normalized_acoustic+phoneme_onset+words',
          'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes+words'],
    
    'Phoneme Surp./Ent. Shuffled': ['Control2_Delta+Theta_STG_sources_normalized_acoustic+phoneme_surprisal_entropy+words_shuffled_new',
            'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes+words']
}


FEATURES = ['Phoneme Surp./Ent.', 'Phoneme Surp./Ent. Shuffeled']
OUTPUT_CSV = 'Dataset1_revision_STG_delta_theta_sources_phoneme_surprisal_entropy_with_words_shuffeled_vs_normal.csv'
OUTPUT_PLOT = 'phoneme_surprisal_entropy_shuffled_vs_normal_boxplot.svg'

# Conditions to analyze
CONDITIONS = ['Sentences', 'Word_list']

# =============================================================================
# MANUAL SIGNIFICANCE SIGNS
# =============================================================================
# Set these manually based on your LME or other statistical results.
# Order:
#   [0] Sentences vs Word_list for first feature (Incremental)
#   [1] Sentences vs Word_list for second feature (Shuffled)
#   [2] Feature 1 vs Feature 2 for Sentences
#   [3] Feature 1 vs Feature 2 for Word_list
# Use: 'n.s.', '*', '**', '***', '****' etc.
SIG_SIGNS = ['n.s.', 'n.s.', 'n.s.', 'n.s.']

# Plotting configuration
FONT = "Times New Roman"

RC_PARAMS = {
    'figure.dpi': 100,
    'savefig.dpi': 100,
    'savefig.transparent': True,
    'font.family': FONT,
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'figure.figsize': (3, 3),
}

pyplot.rcParams.update(RC_PARAMS)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_trf_data(subject, model_name, condition, hemisphere):
    """Load TRF data for a given subject, model, condition, and hemisphere."""
    trf_dir = SUBJECT_DIR / subject / 'meg' / 'TRF' / condition
    trf_file = trf_dir / f'{subject} {model_name}_{hemisphere}.pickle'
    return eelbrain.load.unpickle(trf_file)


def compute_feature_contribution(subject, baseline_model, full_model, condition):
    """
    Compute feature contribution by subtracting baseline model from full model.
    Returns the mean accuracy improvement across both hemispheres.
    """
    # Load baseline models
    trf_lh_baseline = load_trf_data(subject, baseline_model, condition, 'lh')
    trf_rh_baseline = load_trf_data(subject, baseline_model, condition, 'rh')
    
    # Load full models
    trf_lh_full = load_trf_data(subject, full_model, condition, 'lh')
    trf_rh_full = load_trf_data(subject, full_model, condition, 'rh')
    
    # Calculate improvement (full - baseline)
    rh_improvement = (trf_rh_full.proportion_explained.x - 
                     trf_rh_baseline.proportion_explained.x)
    lh_improvement = (trf_lh_full.proportion_explained.x - 
                     trf_lh_baseline.proportion_explained.x)
    
    # Average across sources and hemispheres
    return np.mean([np.mean(rh_improvement), np.mean(lh_improvement)])


def process_subject(subject, condition, baseline_model, full_model, feature):
    """Process a single subject and return accuracy data."""
    print(f"Processing {subject} - {condition} - {feature}")
    
    accuracy = compute_feature_contribution(
        subject, baseline_model, full_model, condition
    )
    
    return {
        'feature': feature,
        'condition': condition,
        'subject': subject,
        'accuracy': accuracy
    }


def collect_all_data(model_comparisons, conditions, subjects):
    """Collect accuracy data for all subjects, conditions, and features."""
    rows = []
    
    for feature_name, (baseline_model, full_model) in model_comparisons.items():
        print(f"\nProcessing feature: {feature_name}")
        
        for condition in conditions:
            print(f"  Condition: {condition}")
            
            for subject in subjects:
                row = process_subject(
                    subject, condition, baseline_model, full_model, feature_name
                )
                rows.append(row)
    
    return pd.DataFrame(rows)


def print_descriptive_statistics(df):
    """Print descriptive statistics for each feature-condition pair."""
    print("\n" + "="*70)
    print("Descriptive Statistics")
    print("="*70)
    
    for feature in df['feature'].unique():
        for condition in df['condition'].unique():
            data = df[(df['feature'] == feature) & 
                     (df['condition'] == condition)]['accuracy']
            
            print(f"\n{feature} - {condition}:")
            print(f"  Mean: {data.mean():.6f}")
            print(f"  SEM: {sem(data):.6f}")
            print(f"  N: {len(data)}")


def add_significance_bars_boxplot(ax, box_data, positions, sig_signs):
    """Add significance indicators to the boxplot."""
    col = 'k'
    h = 0.00008
    
    # Calculate y positions based on data range
    y_max = max([max(data) for data in box_data])
    y_min = min([min(data) for data in box_data])
    y_range = y_max - y_min
    
    y_base = y_max + y_range * 0.05
    y_positions = [
        y_base,                    # First feature: condition 1 vs 2
        y_base,                    # Second feature: condition 1 vs 2
        y_base + y_range * 0.1,    # Condition 1: feature 1 vs 2
        y_base + y_range * 0.18    # Condition 2: feature 1 vs 2
    ]
    
    # Define comparisons: (pos1, pos2, sig_index)
    comparisons = [
        (positions[0], positions[1], 0),  # First feature: condition 1 vs 2
        (positions[2], positions[3], 1),  # Second feature: condition 1 vs 2
        (positions[0], positions[2], 2),  # Condition 1: feature 1 vs 2
        (positions[1], positions[3], 3),  # Condition 2: feature 1 vs 2
    ]
    
    for (x1, x2, sig_idx), y in zip(comparisons, y_positions):
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c=col)
        ax.text((x1+x2)*0.5, y+h/2, sig_signs[sig_idx], 
                ha='center', va='bottom', color=col, fontsize=8)


def plot_boxplots(df, sig_signs, output_path=None):
    """Create boxplot visualization with manually specified significance brackets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with columns: feature, condition, subject, accuracy
    sig_signs : list of str
        Manual significance labels, e.g. ['n.s.', '**', '***', '****']
    output_path : Path or str, optional
        Path to save the plot
    """
    # Update RC params for this plot
    RC_PLOT = {**RC_PARAMS, 'figure.figsize': (3, 3)}
    pyplot.rcParams.update(RC_PLOT)
    
    # Prepare data
    box_data = []
    positions = []
    colors_list = []
    pos_counter = 0
    
    colors = {'Sentences': '#d9a359', 'Word_list': '#2f4858'}
    
    for i, feature in enumerate(FEATURES):
        for j, condition in enumerate(CONDITIONS):
            data = df[(df['feature'] == feature) & 
                     (df['condition'] == condition)].accuracy.values
            box_data.append(data)
            positions.append(pos_counter)
            colors_list.append(colors[condition])
            pos_counter += 1
        pos_counter += 0.5  # Gap between feature groups
    
    # Create plot
    fig, ax = plt.subplots()
    
    # Box plot - caps removed by setting capprops linewidth to 0
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
    
    # Add manually specified significance bars
    add_significance_bars_boxplot(ax, box_data, positions, sig_signs)
    
    # Format plot
    ax.set_xticks([0.5, 3])
    ax.set_xticklabels(['Incremental', 'Shuffled'], fontsize=8)
    ax.set_ylabel('Accuracy improvement $\mathregular{R^{2}}$', fontsize=8)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_title('Sentences vs Word List', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    
    # Adjust y-axis to show all brackets
    y_max = max([max(data) for data in box_data])
    y_min = min([min(data) for data in box_data])
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.25)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['Sentences'], edgecolor='black', 
              label='Sentences', alpha=0.7),
        Patch(facecolor=colors['Word_list'], edgecolor='black', 
              label='Word List', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='svg')
        print(f"\nPlot saved to: {output_path}")
    
    plt.show()
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main analysis workflow."""
    print("="*70)
    print(f"TRF ACCURACY ANALYSIS: {ANALYSIS_TYPE.upper().replace('_', ' ')}")
    print("SHUFFLED VS NORMAL COMPARISON")
    print("="*70)
    
    # Collect all data
    print("\nCOLLECTING DATA")
    print("="*70)
    # df = collect_all_data(MODEL_COMPARISONS, CONDITIONS, SUBJECTS)
    
    # Save data
    output_csv = RESULTS_PATH / OUTPUT_CSV
    
    df = pd.read_csv(output_csv)
    # df.to_csv(output_csv, index=False)
    print(f"\nData saved to: {output_csv}")
    
    # Descriptive statistics (no hypothesis tests)
    print_descriptive_statistics(df)
    
    # Create visualization with manually specified significance
    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)
    output_plot = RESULTS_PATH / OUTPUT_PLOT
    fig = plot_boxplots(df, SIG_SIGNS, output_plot)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return df


# =============================================================================
# Execute
# =============================================================================

if __name__ == "__main__":
    df = main()