#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: filiztezcan

This script calculates the explained variance of acoustic features by subtracting 
the phoneme features model from the full model for both Words and Syllables conditions.
Supports both Dutch and Chinese participant datasets.
"""

import os
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd
import scipy
from scipy.stats import sem
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from cycler import cycler
from matplotlib.ticker import MaxNLocator

import mne
import eelbrain


# =============================================================================
# CONFIGURATION
# =============================================================================

# Choose dataset: 'Dutch_participants' or 'Chinese_participants'
DATASET = 'Dutch_participants'  # Change to 'Chinese_participants' for Chinese participants
# Paths
root = Path.cwd().parents[1]
SUBJECT_DIR = root / DATASET  / 'processed' 
RESULTS_PATH = root / 'Scripts' / 'Accuracy_analysis' / 'Results_publication'

    

if DATASET == 'Dutch_participants':
    
    # Dutch participants
    SUBJECTS = [
        'sub-003', 'sub-005', 'sub-007', 'sub-008', 'sub-009',
        'sub-010', 'sub-012', 'sub-013', 'sub-014', 'sub-015',
        'sub-016', 'sub-017', 'sub-018', 'sub-019', 'sub-020'
    ]
    PARTICIPANT_GROUP = 'Dutch Participants'
    NATIVE_STIMULI = 'Dutch_stimuli'
    NONNATIVE_STIMULI = 'Chinese_stimuli'
    
else:  # Chinese

    # Chinese participants
    SUBJECTS = [
        'sub-021', 'sub-022', 'sub-023', 'sub-024', 'sub-025',
        'sub-026', 'sub-028', 'sub-029', 'sub-032', 'sub-033', 
        'sub-034', 'sub-027', 'sub-030', 'sub-035'
    ]
    PARTICIPANT_GROUP = 'Chinese Participants'
    NATIVE_STIMULI = 'Chinese_stimuli'
    NONNATIVE_STIMULI = 'Dutch_stimuli'

# Model definitions
MODELS_DUTCH = [
    ['Control2_Delta+Theta_STG_sources_normalized_Dutch_stimuli_phonemes',
     'Control2_Delta+Theta_STG_sources_normalized_Dutch_stimuli_acoustic+phonemes'],
    ['Control2_Delta+Theta_STG_sources_normalized_Dutch_stimuli_acoustic',
     'Control2_Delta+Theta_STG_sources_normalized_Dutch_stimuli_acoustic+phonemes']
]

MODELS_CHINESE = [
    ['Control2_Delta+Theta_STG_sources_normalized_Chinese_stimuli_phonemes',
     'Control2_Delta+Theta_STG_sources_normalized_Chinese_stimuli_acoustic+phonemes'],
    ['Control2_Delta+Theta_STG_sources_normalized_Chinese_stimuli_acoustic',
     'Control2_Delta+Theta_STG_sources_normalized_Chinese_stimuli_acoustic+phonemes']
]

FEATURES = ['Acoustic Edge', 'Phoneme Onset']

# Plotting configuration
FONT = "Times New Roman"
COLORS_CYCLE = cycler(color=['lightgrey', 'dimgrey', 'yellowgreen', 
                              'olivedrab', 'coral', 'orangered'])
RC_PARAMS = {
    'figure.dpi': 100,
    'savefig.dpi': 100,
    'savefig.transparent': True,
    'font.family': FONT,
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'figure.figsize': (3, 3),
    'axes.prop_cycle': COLORS_CYCLE
}

pyplot.rcParams.update(RC_PARAMS)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_trf_data(subject, condition, model_name, hemisphere):
    """Load TRF data for a given subject, condition, model, and hemisphere."""
    trf_dir = SUBJECT_DIR / subject / 'meg' / 'TRF' / condition
    filepath = trf_dir / f'{subject} {model_name}_{hemisphere}.pickle'
    return eelbrain.load.unpickle(filepath)


def calculate_accuracy_difference(subject, condition, models, hemisphere):
    """Calculate the difference in proportion explained between two models."""
    trf_baseline = load_trf_data(subject, condition, models[0], hemisphere)
    trf_full = load_trf_data(subject, condition, models[1], hemisphere)
    
    return trf_full.proportion_explained.x - trf_baseline.proportion_explained.x


def process_subject(subject, condition, models, feature, stimuli_type):
    """Process a single subject and return accuracy data."""
    print(f"Processing {subject} - {condition} - {feature} - {stimuli_type}")
    
    # Calculate differences for both hemispheres
    diff_lh = calculate_accuracy_difference(subject, condition, models, 'lh')
    diff_rh = calculate_accuracy_difference(subject, condition, models, 'rh')
    
    # Average across hemispheres and sources
    accuracy = np.mean([np.mean(diff_rh), np.mean(diff_lh)])
    
    return {
        'Stimuli': stimuli_type,
        'Condition': condition,
        'Feature': feature,
        'subject': subject,
        'accuracy': accuracy
    }


def collect_all_data():
    """Collect accuracy data for all subjects, conditions, and features."""
    rows = []
    
    # Iterate over features (Acoustic Edge, Phoneme Onset)
    for feature_idx in range(2):
        models_dutch = MODELS_DUTCH[feature_idx]
        models_chinese = MODELS_CHINESE[feature_idx]
        feature = FEATURES[feature_idx]
        
        # Iterate over conditions
        for condition in ['Words', 'Syllables']:
            # Dutch stimuli
            for subject in SUBJECTS:
                row = process_subject(subject, condition, models_dutch, 
                                     feature, 'Dutch_stimuli')
                rows.append(row)
            
            # Chinese stimuli
            for subject in SUBJECTS:
                row = process_subject(subject, condition, models_chinese, 
                                     feature, 'Chinese_stimuli')
                rows.append(row)
    
    return pd.DataFrame(rows)


def run_statistical_tests(df, feature, filter1, filter2):
    """Run t-tests comparing two groups."""
    group1 = df[filter1]['accuracy'].tolist()
    group2 = df[filter2]['accuracy'].tolist()
    
    test1 = scipy.stats.ttest_1samp(group1, 0)
    test2 = scipy.stats.ttest_1samp(group2, 0)
    
    return test1, test2


def add_significance_bars_boxplot(ax, box_data, positions, sig_signs):
    """Add significance indicators to the boxplot."""
    col = 'k'
    h = 0.00008
    
    # Calculate y positions based on max values in each comparison
    y_base = 0.0055
    y_positions = [y_base, y_base, y_base + 0.00055, y_base + 0.001]
    
    # Define comparisons: (pos1, pos2, sig_index)
    comparisons = [
        (positions[0], positions[1], 0),  # First feature: condition 1 vs 2
        (positions[2], positions[3], 3),  # Second feature: condition 1 vs 2
        (positions[0], positions[2], 2),  # Condition 1: feature 1 vs 2
        (positions[1], positions[3], 1),  # Condition 2: feature 1 vs 2
    ]
    
    for (x1, x2, sig_idx), y in zip(comparisons, y_positions):
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c=col)
        plt.text((x1+x2)*0.5, y+h/2, sig_signs[sig_idx], 
                ha='center', va='bottom', color=col, fontsize=8)


def plot_boxplot_by_stimuli(df, stimuli_type, sig_signs):
    """Create box plot comparing conditions for a specific stimuli type."""
    df_filtered = df[df.Stimuli == stimuli_type]
    
    # Prepare data
    box_data = []
    positions = []
    colors_list = []
    pos_counter = 0
    
    colors = {'Words': '#2f4858', 'Syllables': '#719e87'}
    
    for i, feature in enumerate(FEATURES):
        for j, condition in enumerate(['Words', 'Syllables']):
            data = df_filtered[(df_filtered.Feature == feature) & 
                              (df_filtered.Condition == condition)].accuracy.values
            box_data.append(data)
            positions.append(pos_counter)
            colors_list.append(colors[condition])
            pos_counter += 1
        pos_counter += 0.5  # Gap between feature groups
    
    # Create plot
    RC_PLOT = {**RC_PARAMS, 'figure.figsize': (3, 3)}
    pyplot.rcParams.update(RC_PLOT)
    
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
    
    # Add significance bars
    add_significance_bars_boxplot(ax, box_data, positions, sig_signs)
    
    # Format plot
    ax.set_xticks([0.5, 3])
    ax.set_xticklabels(['Acoustic Edge', 'Phonemes'], fontsize=8)
    ax.set_ylabel('Accuracy improvement $\mathregular{R^{2}}$', fontsize=12)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_title(f'{PARTICIPANT_GROUP} {stimuli_type}', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylim(-0.0003, 0.007)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['Words'], edgecolor='black', 
                            label='Words', alpha=0.7),
                      Patch(facecolor=colors['Syllables'], edgecolor='black', 
                            label='Syllables', alpha=0.7)]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_boxplot_by_condition(df, condition, sig_signs, DATASET):
    """Create box plot comparing native vs nonnative stimuli for a specific condition."""
    df_filtered = df[df.Condition == condition]
    
    # Prepare data
    box_data = []
    positions = []
    colors_list = []
    pos_counter = 0
    

    if DATASET == 'Dutch_participants':
        colors = {'Native': '#504538', 'Nonnative': '#e39ba6'}
    else:
        colors = {'Native': '#504538', 'Nonnative': '#9e9385'}
    

    
    # Order: Native first, then Nonnative
    stimuli_order = [
        (NATIVE_STIMULI, 'Native'),
        (NONNATIVE_STIMULI, 'Nonnative')
    ]
    
    for i, feature in enumerate(FEATURES):
        for stimuli_key, stimuli_label in stimuli_order:
            data = df_filtered[(df_filtered.Feature == feature) & 
                              (df_filtered.Stimuli == stimuli_key)].accuracy.values
            box_data.append(data)
            positions.append(pos_counter)
            colors_list.append(colors[stimuli_label])
            pos_counter += 1
        pos_counter += 0.5  # Gap between feature groups
    
    # Create plot
    RC_PLOT = {**RC_PARAMS, 'figure.figsize': (3, 3)}
    pyplot.rcParams.update(RC_PLOT)
    
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
    
    # Add significance bars
    add_significance_bars_boxplot(ax, box_data, positions, sig_signs)
    
    # Format plot
    ax.set_xticks([0.5, 3])
    ax.set_xticklabels(['Acoustic Edge', 'Phonemes'], fontsize=8)
    ax.set_ylabel('Accuracy improvement $\mathregular{R^{2}}$', fontsize=12)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_title(f'{PARTICIPANT_GROUP} {condition}', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylim(-0.0003, 0.007)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['Native'], edgecolor='black', 
                            label='Native', alpha=0.7),
                      Patch(facecolor=colors['Nonnative'], edgecolor='black', 
                            label='Nonnative', alpha=0.7)]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def print_statistics(df, test_name, feature, filter1, filter2, label1, label2):
    """Print statistical test results in a formatted way."""
    test1, test2 = run_statistical_tests(df, feature, filter1, filter2)
    
    print(f"\n{test_name} - {feature}:")
    print(f"  {label1}: t={test1.statistic:.4f}, p={test1.pvalue:.4e}")
    print(f"  {label2}: t={test2.statistic:.4f}, p={test2.pvalue:.4e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Collect all data
    print("=" * 70)
    print(f"COLLECTING DATA FOR {PARTICIPANT_GROUP}")
    print("=" * 70)
    df = collect_all_data()
    
    # Optional: Save data
    output_file = RESULTS_PATH / f'{DATASET}_Accuracies_STG_sources_all_stimuli.csv'
    df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")
    
    # =============================================================================
    # Analysis 1: By stimuli type (Dutch vs Chinese)
    # =============================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: BY STIMULI TYPE")
    print("=" * 70)
    
    for stimuli in ['Dutch_stimuli', 'Chinese_stimuli']:
        print(f"\n{'-'*70}")
        print(f"Stimuli: {stimuli}")
        print(f"{'-'*70}")
        
        # Significance signs - adjust based on your actual statistical results
        # Order: [Words vs Syllables for Acoustic, Words vs Syllables for Phoneme,
        #         Acoustic vs Phoneme for Words, Acoustic vs Phoneme for Syllables]
        if DATASET == 'Dutch_participants':
            if stimuli == 'Dutch_stimuli':
                sig_signs = ['n.s.', 'n.s.', '*', '*****']
            else:             
                sig_signs = ['n.s.', '***', '**', '**']
        else:  # Chinese participants
            if stimuli == 'Dutch_stimuli':
                sig_signs = ['n.s.', '**', 'n.s.', '****']
            else:
                sig_signs = ['n.s.', '**', 'n.s.', '****']
        
        # Create plot
        fig = plot_boxplot_by_stimuli(df, stimuli, sig_signs)
        
        # Save figure
        fig_name = f'{DATASET}_Accuracy_boxplot_{stimuli}.svg'
        plt.savefig(RESULTS_PATH / fig_name, dpi=300, bbox_inches='tight', format='svg')
        
        # Statistical tests
        df_subset = df[df.Stimuli == stimuli]
        
        for feature in FEATURES:
            filter1 = (df_subset['Feature'] == feature) & (df_subset['Condition'] == 'Words')
            filter2 = (df_subset['Feature'] == feature) & (df_subset['Condition'] == 'Syllables')
            print_statistics(df_subset, stimuli, feature, filter1, filter2, 
                           'Words', 'Syllables')
    
    # =============================================================================
    # Analysis 2: By condition (Words vs Syllables) - Native vs Nonnative
    # =============================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: BY CONDITION (Native vs Nonnative)")
    print("=" * 70)
    
    for condition in ['Words']:
        print(f"\n{'-'*70}")
        print(f"Condition: {condition}")
        print(f"Native stimuli: {NATIVE_STIMULI}")
        print(f"Nonnative stimuli: {NONNATIVE_STIMULI}")
        print(f"{'-'*70}")
        
        # Significance signs - adjust based on your actual statistical results
        # Order: [Native vs Nonnative for Acoustic, Native vs Nonnative for Phoneme,
        #         Acoustic vs Phoneme for Native, Acoustic vs Phoneme for Nonnative]
        if DATASET == 'Dutch_participants':
            sig_signs = ['n.s.', 'n.s.', '*', '**']
        else:  # Chinese participants
            sig_signs = ['n.s.', 'n.s.', 'n.s.', 'n.s.']
        
        # Create plot
        fig = plot_boxplot_by_condition(df, condition, sig_signs,DATASET)
        
        # Save figure
        fig_name = f'{DATASET}_Accuracy_boxplot_{condition}.svg'
        plt.savefig(RESULTS_PATH / fig_name, dpi=300, bbox_inches='tight', format='svg')
        
        # Statistical tests
        df_subset = df[df.Condition == condition]
        
        for feature in FEATURES:
            filter1 = (df_subset['Feature'] == feature) & (df_subset['Stimuli'] == NATIVE_STIMULI)
            filter2 = (df_subset['Feature'] == feature) & (df_subset['Stimuli'] == NONNATIVE_STIMULI)
            print_statistics(df_subset, condition, feature, filter1, filter2,
                           'Native', 'Nonnative')
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)