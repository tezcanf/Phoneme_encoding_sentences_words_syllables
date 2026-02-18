"""
TRF Analysis and Comparison Script
Compares TRF weights between Sentences and Word_list conditions across subjects.
Generates visualizations of power of weights for different predictor features.
"""

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, sem
from matplotlib.ticker import MaxNLocator
import eelbrain

# ============================================================================
# Configuration
# ============================================================================
# Paths
DATA_ROOT = Path.cwd().parents[1]
SUBJECTS_DIR = DATA_ROOT / 'processed'
ANOVA_RESULTS_PATH = DATA_ROOT / 'Scripts' / 'TRF_weight_analysis' / 'Output'
OUTPUT_FOLDER = DATA_ROOT / 'Scripts' / 'Visualize_TRF_weights/' / 'Output'


# Analysis parameters
TEST_NAME = 'Control2_Sentence_vs_words'
MODEL_NAME = 'Control2_Delta+Theta_STG_sources_normalized_acoustic+phonemes+words'
CONDITIONS = ['Sentences', 'Word_list']

# Subjects
SUBJECTS = [
    'sub-002', 'sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007',
    'sub-008', 'sub-009', 'sub-010', 'sub-011', 'sub-012', 'sub-013',
    'sub-014', 'sub-016', 'sub-017', 'sub-018', 'sub-019', 'sub-020',
    'sub-021'
]

# Plotting parameters
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

# Colors for conditions
COLOR_SENTENCES = '#d9a359'  # Gold/orange for Sentences
COLOR_WORDLIST = '#2f4858'   # Dark blue for Word_list


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_trfs_for_hemisphere(condition, hemisphere, subjects, model_name):
    """
    Load TRFs for all subjects in a given condition and hemisphere.
    
    Parameters:
    -----------
    condition : str
        'Sentences' or 'Word_list'
    hemisphere : str
        'lh' or 'rh'
    subjects : list
        List of subject IDs
    model_name : str
        Name of the TRF model
        
    Returns:
    --------
    dataset : eelbrain.Dataset
        Dataset containing TRFs for all subjects
    """
    rows = []
    x_names = None
    
    for subject in subjects:
        print(f"Loading {condition} - {hemisphere} - {subject}")
        trf_dir = SUBJECTS_DIR / subject / 'meg' / 'TRF' / condition
        trf_path = trf_dir / f'{subject} {model_name}_{hemisphere}.pickle'
        
        trf = eelbrain.load.unpickle(trf_path)
        trf.r.source._subjects_dir = SUBJECTS_DIR
        
        rows.append([subject, trf.r, *trf.h])
        x_names = trf.x
    
    dataset = eelbrain.Dataset.from_caselist(['subject', 'det', *x_names], rows)
    return dataset


def load_all_trfs(subjects, model_name):
    """
    Load TRFs for all subjects, conditions, and hemispheres.
    
    Returns:
    --------
    dict : Dictionary with keys for each condition and hemisphere
    """
    data = {}
    
    for condition in CONDITIONS:
        for hemisphere in ['lh', 'rh']:
            key = f"{condition}_{hemisphere}"
            data[key] = load_trfs_for_hemisphere(condition, hemisphere, subjects, model_name)
    
    return data


# ============================================================================
# TRF Power Computation Functions
# ============================================================================

def compute_acoustic_edge_power(data_lh, data_rh):
    """
    Compute power of weights for acoustic edge features.
    Averages across frequency and source dimensions, then across hemispheres.
    """
    lh_power = data_lh["gammatone_on"].square().sqrt().mean('frequency').mean('source')
    rh_power = data_rh["gammatone_on"].square().sqrt().mean('frequency').mean('source')
    return (lh_power + rh_power) / 2


def compute_phoneme_feature_power(data_lh, data_rh):
    """
    Compute power of weights for phoneme features.
    Averages across phoneme onset, surprisal, and entropy, then across hemispheres.
    """
    # Left hemisphere
    lh_onset = data_lh["phonemes"].square().sqrt().mean('source')
    lh_surprisal = data_lh["cohort_surprisal"].square().sqrt().mean('source')
    lh_entropy = data_lh["cohort_entropy"].square().sqrt().mean('source')
    
    # Right hemisphere
    rh_onset = data_rh["phonemes"].square().sqrt().mean('source')
    rh_surprisal = data_rh["cohort_surprisal"].square().sqrt().mean('source')
    rh_entropy = data_rh["cohort_entropy"].square().sqrt().mean('source')
    
    # Average across all features and hemispheres
    return (lh_onset + lh_surprisal + lh_entropy + 
            rh_onset + rh_surprisal + rh_entropy) / 6


def compute_all_feature_powers(trf_data):
    """
    Compute power of weights for all features across conditions.
    
    Returns:
    --------
    dict : Dictionary containing power values and significance masks for each feature
    """
    features = []
    
    # Acoustic edges
    acoustic_sentences = compute_acoustic_edge_power(
        trf_data['Sentences_lh'], trf_data['Sentences_rh']
    )
    acoustic_wordlist = compute_acoustic_edge_power(
        trf_data['Word_list_lh'], trf_data['Word_list_rh']
    )
    acoustic_p = np.load(
        os.path.join(ANOVA_RESULTS_PATH, f'acoustic_edge_{TEST_NAME}_source_STG.npy'),
        allow_pickle=True
    )
    features.append({
        'title': 'Acoustic Edges',
        'sentences': acoustic_sentences,
        'wordlist': acoustic_wordlist,
        'p_values': acoustic_p
    })
    
    # Phoneme features
    phoneme_sentences = compute_phoneme_feature_power(
        trf_data['Sentences_lh'], trf_data['Sentences_rh']
    )
    phoneme_wordlist = compute_phoneme_feature_power(
        trf_data['Word_list_lh'], trf_data['Word_list_rh']
    )
    phoneme_p = np.load(
        os.path.join(ANOVA_RESULTS_PATH, f'phoneme_onset_{TEST_NAME}_source_STG.npy'),
        allow_pickle=True
    )
    features.append({
        'title': 'Phoneme Features',
        'sentences': phoneme_sentences,
        'wordlist': phoneme_wordlist,
        'p_values': phoneme_p
    })
    
    return features


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_feature_comparison(ax, feature_data, color_sentences=COLOR_SENTENCES, 
                           color_wordlist=COLOR_WORDLIST):
    """
    Plot comparison of a single feature across conditions on given axis.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    feature_data : dict
        Dictionary containing sentences/wordlist data and p-values
    """
    # Extract data
    sentences = feature_data['sentences']
    wordlist = feature_data['wordlist']
    p_values = feature_data['p_values']
    title = feature_data['title']
    
    # Trim edge artifacts (first 2 and last 2 timepoints)
    sentences_mean = sentences.mean('case').x[2:-2]
    wordlist_mean = wordlist.mean('case').x[2:-2]
    time = sentences.time.times[2:-2]
    
    # Compute standard error
    sentences_error = [sem(sentences.x[:, t]) for t in range(len(time))]
    wordlist_error = [sem(wordlist.x[:, t]) for t in range(len(time))]
    
    # Plot sentences condition
    ax.plot(time, sentences_mean, color_sentences, linewidth=0.5, label='Sentences')
    ax.fill_between(time, sentences_mean - sentences_error, 
                    sentences_mean + sentences_error,
                    alpha=0.7, color=color_sentences, linewidth=0.0)
    
    # Plot word list condition
    ax.plot(time, wordlist_mean, color_wordlist, linewidth=0.5, label='Word list')
    ax.fill_between(time, wordlist_mean - wordlist_error,
                    wordlist_mean + wordlist_error,
                    alpha=0.7, color=color_wordlist, linewidth=0.0)
    
    # Plot significance markers
    ax.plot(time, np.multiply(p_values, -0.00), 'red', linewidth=2)
    
    # Styling
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title(title)


def create_comparison_figure(features, output_path=None):
    """
    Create complete comparison figure for all features.
    
    Parameters:
    -----------
    features : list
        List of feature dictionaries
    output_path : str or Path, optional
        Path to save figure. If None, figure is displayed but not saved.
    """
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(2.8, 3),
                            sharex=True, sharey=True)
    
    # Handle case of single feature
    if n_features == 1:
        axes = [axes]
    
    # Plot each feature
    for i, feature in enumerate(features):
        plot_feature_comparison(axes[i], feature)
    
    # Adjust spacing
    plt.subplots_adjust(hspace=0.0)
    
    # Add axis labels
    fig.text(0.5, 0., 'Time (sec)', ha='center', fontsize=12)
    fig.text(0, 0.5, 'Power of Weights $\\mathregular{\\sqrt{w^{2}}}$',
             va='center', rotation='vertical', fontsize=12)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',format='svg')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()


# ============================================================================
# Statistical Analysis Functions
# ============================================================================

def perform_statistical_tests(features):
    """
    Perform paired t-tests comparing conditions for each feature.
    
    Returns:
    --------
    dict : Dictionary with test results for each feature
    """
    results = {}
    
    for feature in features:
        title = feature['title']
        sentences = feature['sentences'].x
        wordlist = feature['wordlist'].x
        
        # Perform paired t-test at each timepoint
        t_stats = []
        p_vals = []
        
        for t in range(sentences.shape[1]):
            t_stat, p_val = ttest_rel(sentences[:, t], wordlist[:, t])
            t_stats.append(t_stat)
            p_vals.append(p_val)
        
        results[title] = {
            't_statistics': np.array(t_stats),
            'p_values': np.array(p_vals),
            'significant_times': feature['sentences'].time.times[np.array(p_vals) < 0.05]
        }
    
    return results


def print_statistical_summary(stat_results):
    """Print summary of statistical test results."""
    print("\n" + "="*80)
    print("Statistical Test Results Summary")
    print("="*80)
    
    for feature_name, results in stat_results.items():
        print(f"\n{feature_name}:")
        print(f"  Number of significant timepoints (p < 0.05): "
              f"{np.sum(results['p_values'] < 0.05)}")
        print(f"  Minimum p-value: {np.min(results['p_values']):.6f}")
        
        if len(results['significant_times']) > 0:
            print(f"  Time range with significance: "
                  f"{results['significant_times'][0]:.3f} - "
                  f"{results['significant_times'][-1]:.3f} s")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("TRF Comparison Analysis: Sentences vs Word List")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print(f"Number of subjects: {len(SUBJECTS)}")
    print(f"Conditions: {CONDITIONS}")
    print("="*80)
    
    # Load TRF data
    print("\nLoading TRF data...")
    trf_data = load_all_trfs(SUBJECTS, MODEL_NAME)
    
    # Compute feature powers
    print("\nComputing feature powers...")
    features = compute_all_feature_powers(trf_data)
    
    # Perform statistical tests
    print("\nPerforming statistical tests...")
    stat_results = perform_statistical_tests(features)
    print_statistical_summary(stat_results)
    
    # Create visualization
    print("\nCreating visualization...")
    
    output_path = os.path.join(OUTPUT_FOLDER,'Dutch_sentences_vs_words_TRF_visualization.svg')
    create_comparison_figure(features,output_path)
    
    print("\n" + "="*80)
    print("Analysis completed!")
    print("="*80)


if __name__ == "__main__":
    main()