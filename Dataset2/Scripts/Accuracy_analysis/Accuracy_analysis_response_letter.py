#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: filiztezcan

This script calculates the explained variance of acoustic features by subtracting 
the phoneme features model from the full model for both Words and Syllables conditions,
comparing Dutch and Chinese stimuli across Dutch and Chinese participants.
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
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from cycler import cycler

import eelbrain
import mne

# ============================================================================
# CONFIGURATION
# ============================================================================


# Paths
root = Path.cwd().parents[1]

RESULTS_PATH = root / 'Scripts' / 'Accuracy_analysis' / 'Results_publication_revision'
DUTCH_DATA_ROOT = root / 'Dutch_participants'  / 'processed' 
CHINESE_DATA_ROOT = root / 'Chinese_participants'  / 'processed' 

# Subjects
DUTCH_SUBJECTS = [
    'sub-003', 'sub-005', 'sub-007', 'sub-008', 'sub-009',
    'sub-010', 'sub-012', 'sub-013', 'sub-014', 'sub-015',
    'sub-016', 'sub-017', 'sub-018', 'sub-019', 'sub-020'
]

CHINESE_SUBJECTS = [
    'sub-021', 'sub-022', 'sub-023', 'sub-024', 'sub-025',
    'sub-026', 'sub-027', 'sub-028', 'sub-029', 'sub-030',
    'sub-032', 'sub-033', 'sub-034', 'sub-035'
]

# Model definitions
MODELS_DUTCH = [
    ['Control2_Delta+Theta_STG_sources_normalized_Dutch_stimuli_phoneme_surp_ent',
     'Control2_Delta+Theta_STG_sources_normalized_Dutch_stimuli_acoustic+phonemes'],
    ['Control2_Delta+Theta_STG_sources_normalized_Dutch_stimuli_phoneme_onset',
     'Control2_Delta+Theta_STG_sources_normalized_Dutch_stimuli_acoustic+phonemes']
]

MODELS_CHINESE = [
    ['Control2_Delta+Theta_STG_sources_normalized_Chinese_stimuli_phoneme_surp_ent',
     'Control2_Delta+Theta_STG_sources_normalized_Chinese_stimuli_acoustic+phonemes'],
    ['Control2_Delta+Theta_STG_sources_normalized_Chinese_stimuli_phoneme_onset',
     'Control2_Delta+Theta_STG_sources_normalized_Chinese_stimuli_acoustic+phonemes']
]

FEATURES = ['Phoneme Onset', 'Phoneme Surprisal/Entropy']
CONDITIONS = ['Words', 'Syllables']
STIMULI_TYPES = ['Dutch_stimuli', 'Chinese_stimuli']

# Plotting configuration
FONT = "Times New Roman"
COLORS = cycler(color=['lightgrey', 'dimgrey', 'yellowgreen', 
                       'olivedrab', 'coral', 'orangered'])
RC = {
    'figure.dpi': 100,
    'savefig.dpi': 100,
    'savefig.transparent': True,
    'font.family': FONT,
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'figure.figsize': (3, 3),
    'axes.prop_cycle': COLORS
}



pyplot.rcParams.update(RC)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_trf_data(data_root, subject, condition, model_name, hemisphere):
    """Load TRF data for a specific subject, condition, model, and hemisphere."""
    trf_dir = data_root / subject / 'meg' / 'TRF' / condition
    filepath = trf_dir / f'{subject} {model_name}_{hemisphere}.pickle'
    return eelbrain.load.unpickle(filepath)


def calculate_accuracy_improvement(data_root, subjects, models, condition, 
                                   stimuli_type, feature_name):
    """
    Calculate accuracy improvement by subtracting baseline model from full model.
    
    Parameters
    ----------
    data_root : Path
        Root directory for data
    subjects : list
        List of subject IDs
    models : list
        [baseline_model, full_model] names
    condition : str
        'Words' or 'Syllables'
    stimuli_type : str
        'Dutch_stimuli' or 'Chinese_stimuli'
    feature_name : str
        Name of the feature being analyzed
        
    Returns
    -------
    list
        List of [stimuli_type, condition, feature, subject, accuracy] rows
    """
    rows = []
    
    for subject in subjects:
        print(f"Processing {subject}")
        
        # Load baseline model (phoneme features only)
        trf_lh0 = load_trf_data(data_root, subject, condition, models[0], 'lh')
        trf_rh0 = load_trf_data(data_root, subject, condition, models[0], 'rh')
        
        # Load full model (acoustic + phoneme features)
        trf_lh = load_trf_data(data_root, subject, condition, models[1], 'lh')
        trf_rh = load_trf_data(data_root, subject, condition, models[1], 'rh')
        
        # Calculate improvement (full - baseline)
        trf_rh.proportion_explained.x -= trf_rh0.proportion_explained.x
        trf_lh.proportion_explained.x -= trf_lh0.proportion_explained.x
        
        # Average across hemispheres and sources
        accuracy = np.mean([
            trf_rh.proportion_explained.mean(axis='source'),
            trf_lh.proportion_explained.mean(axis='source')
        ])
        
        rows.append([stimuli_type, condition, feature_name, subject, accuracy])
    
    return rows


def create_dataframe(data_root, subjects, models_dutch, models_chinese):
    """
    Create complete dataframe with all conditions and features.
    
    Parameters
    ----------
    data_root : Path
        Root directory for data
    subjects : list
        List of subject IDs
    models_dutch : list
        List of model pairs for Dutch stimuli
    models_chinese : list
        List of model pairs for Chinese stimuli
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Stimuli, condition, feature, subject, accuracy
    """
    all_rows = []
    
    for feature_idx in range(2):  # 2 features: Onset and Surprisal/Entropy
        for condition in CONDITIONS:
            # Dutch stimuli
            rows = calculate_accuracy_improvement(
                data_root, subjects, models_dutch[feature_idx], 
                condition, 'Dutch_stimuli', FEATURES[feature_idx]
            )
            all_rows.extend(rows)
            
            # Chinese stimuli
            rows = calculate_accuracy_improvement(
                data_root, subjects, models_chinese[feature_idx],
                condition, 'Chinese_stimuli', FEATURES[feature_idx]
            )
            all_rows.extend(rows)
    
    return pd.DataFrame(
        data=all_rows, 
        columns=['Stimuli', 'condition', 'feature', 'subject', 'accuracy']
    )


def add_significance_bracket(ax, x1, x2, y, h, text, col='k'):
    """
    Add a significance bracket to the plot.
    
    Parameters
    ----------
    ax : matplotlib axis
        The axis to draw on
    x1, x2 : float
        Start and end x-coordinates
    y : float
        Y-coordinate for the bracket
    h : float
        Height of the bracket
    text : str
        Significance text (e.g., '****', 'n.s.')
    col : str
        Color of the bracket
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c=col)
    ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color=col)


def plot_boxplot_with_significance(df, stimuli_type, participant_group, sig_signs, save_path=None):
    """
    Create box plot with significance markers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to plot
    stimuli_type : str
        'Dutch_stimuli' or 'Chinese_stimuli'
    participant_group : str
        'Dutch Participants' or 'Chinese Participants'
    sig_signs : list
        List of significance markers (e.g., ['n.s.', '****'])
        Order: [Onset Words vs Syllables, Surp/Ent Words vs Syllables, 
                Words Onset vs Surp/Ent, Syllables Onset vs Surp/Ent]
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    """
    pyplot.rcParams.update(RC)
    
    # Define colors
    custom_colors = ['#2f4858', '#719e87']  # Words, Syllables
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Create box plot
    box_width = 0.35
    positions_onset = [0 - box_width/2, 0 + box_width/2]
    positions_surp = [1 - box_width/2, 1 + box_width/2]
    
    # Prepare data for boxplot
    data_onset_words = df[(df.feature == 'Phoneme Onset') & 
                          (df.condition == 'Words')]['accuracy'].values
    data_onset_syllables = df[(df.feature == 'Phoneme Onset') & 
                              (df.condition == 'Syllables')]['accuracy'].values
    data_surp_words = df[(df.feature == 'Phoneme Surprisal/Entropy') & 
                         (df.condition == 'Words')]['accuracy'].values
    data_surp_syllables = df[(df.feature == 'Phoneme Surprisal/Entropy') & 
                             (df.condition == 'Syllables')]['accuracy'].values
    
    # Create box plots
    bp1 = ax.boxplot([data_onset_words, data_onset_syllables],
                      positions=positions_onset,
                      widths=box_width * 0.6,
                      patch_artist=True,
                      showfliers=False,
                      boxprops=dict(linewidth=0.5),
                      whiskerprops=dict(linewidth=0.5),
                      capprops=dict(linewidth=0),  # Set cap linewidth to 0
                      medianprops=dict(linewidth=0.5, color='black'))
    
    bp2 = ax.boxplot([data_surp_words, data_surp_syllables],
                      positions=positions_surp,
                      widths=box_width * 0.6,
                      patch_artist=True,
                      showfliers=False,
                      boxprops=dict(linewidth=0.5),
                      whiskerprops=dict(linewidth=0.5),
                      capprops=dict(linewidth=0),  # Set cap linewidth to 0
                      medianprops=dict(linewidth=0.5, color='black'))
    
    # Color the boxes
    for patch, color in zip(bp1['boxes'], custom_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    
    for patch, color in zip(bp2['boxes'], custom_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    
    # Get y-axis limits for positioning significance brackets
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # Calculate positions for significance brackets
    # Find max values for positioning brackets above the data
    max_vals = [
        data_onset_words.max(),
        data_onset_syllables.max(),
        data_surp_words.max(),
        data_surp_syllables.max()
    ]
    top_val = max(max_vals)
    
    h = y_range * 0.01  # Height of bracket
    
    # Bracket 1: Phoneme Onset (Words vs Syllables)
    y1 = top_val + y_range * 0.05
    add_significance_bracket(ax, positions_onset[0], positions_onset[1], 
                            y1, h, sig_signs[0])
    
    # Bracket 2: Phoneme Surprisal/Entropy (Words vs Syllables)
    y2 = top_val + y_range * 0.05
    add_significance_bracket(ax, positions_surp[0], positions_surp[1], 
                            y2, h, sig_signs[1])
    
    # Bracket 3: Words (Onset vs Surprisal)
    y3 = top_val + y_range * 0.15
    add_significance_bracket(ax, positions_onset[0], positions_surp[0], 
                            y3, h, sig_signs[2])
    
    # Bracket 4: Syllables (Onset vs Surprisal)
    y4 = top_val + y_range * 0.25
    add_significance_bracket(ax, positions_onset[1], positions_surp[1], 
                            y4, h, sig_signs[3])
    
    # Format plot
    ax.axhline(y=0, color='black', linestyle='-', lw=0.5, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Phoneme Onset', 'Ph. Surp./Ent.'], fontsize=8)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Accuracy improvement $\mathregular{R^{2}}$', fontsize=12)
    ax.set_title(f'{participant_group} {stimuli_type}', fontsize=12)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=custom_colors[0], edgecolor=custom_colors[0], label='Words'),
        Patch(facecolor=custom_colors[1], edgecolor=custom_colors[1], label='Syllables')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False)
    
    # Format y-axis
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.tick_params(axis='x', which='major', labelsize=10)
    
    # Remove top and right spines
    ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    # plt.close()



def run_statistical_tests(df, stimuli_type):
    """
    Run one-sample t-tests against zero for all conditions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe for specific stimuli type
    stimuli_type : str
        'Dutch_stimuli' or 'Chinese_stimuli'
    """
    print(f"\n{'='*60}")
    print(f"Statistical Tests for {stimuli_type}")
    print(f"{'='*60}\n")
    
    for feature in FEATURES:
        for condition in CONDITIONS:
            mask = (df.feature == feature) & (df.condition == condition)
            data = df[mask]['accuracy'].tolist()
            result = scipy.stats.ttest_1samp(data, 0)
            
            print(f"{feature} - {condition}:")
            print(f"  t-statistic: {result.statistic:.4f}")
            print(f"  p-value: {result.pvalue:.4e}\n")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_participant_group(data_root, subjects, participant_group, 
                              models_dutch, models_chinese):
    """
    Complete analysis pipeline for one participant group.
    
    Parameters
    ----------
    data_root : Path
        Root directory for data
    subjects : list
        List of subject IDs
    participant_group : str
        'Dutch' or 'Chinese'
    models_dutch : list
        Model pairs for Dutch stimuli
    models_chinese : list
        Model pairs for Chinese stimuli
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {participant_group} Participants")
    print(f"{'='*60}\n")
    
    # Create dataframe
    df = create_dataframe(data_root, subjects, models_dutch, models_chinese)
    
    # Optional: Save dataframe
    output_file = os.path.join(
        RESULTS_PATH, 
        f'Dataset2_{participant_group}_Accuracies_STG_sources_all_stimuli_revision.csv'
    )
    df.to_csv(output_file, index=False)
    
    # Significance signs (update based on your actual statistical results)
    sig_signs = ['n.s.', '****', '****', 'n.s.']
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Analyze and plot for each stimuli type
    for stimuli_type in STIMULI_TYPES:
        df_filtered = df[df.Stimuli == stimuli_type]
        
        # Create filename for saving
        filename = f'{participant_group}_participants_{stimuli_type}_boxplot.svg'
        save_path = os.path.join(RESULTS_PATH, filename)
        
        # Plot
        plot_boxplot_with_significance(
            df_filtered, stimuli_type, 
            f'{participant_group} Participants', sig_signs,
            save_path=save_path
        )
        
        # Statistical tests
        run_statistical_tests(df_filtered, stimuli_type)


# ============================================================================
# RUN ANALYSES
# ============================================================================

if __name__ == "__main__":
    # Analyze Dutch participants
    analyze_participant_group(
        DUTCH_DATA_ROOT, DUTCH_SUBJECTS, 'Dutch',
        MODELS_DUTCH, MODELS_CHINESE
    )
    
    # Analyze Chinese participants
    analyze_participant_group(
        CHINESE_DATA_ROOT, CHINESE_SUBJECTS, 'Chinese',
        MODELS_DUTCH, MODELS_CHINESE
    )