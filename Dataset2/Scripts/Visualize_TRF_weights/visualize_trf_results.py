#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merged TRF Visualization Script - Batch Processing

Combines functionality for:
- Language comparison analysis
- Condition comparison (Words vs Syllables) analysis

@author: filiztezcan
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from matplotlib.ticker import MaxNLocator
import eelbrain
import os

# ============================================================================
# MATPLOTLIB CONFIGURATION
# ============================================================================

FONT = "Times New Roman"
RC = {
    'figure.dpi': 100,
    'savefig.dpi': 100,
    'savefig.transparent': True,
    'font.family': FONT,
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 10,
    'figure.figsize': (2.8, 3)
}
plt.rcParams.update(RC)

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

FEATURES = ['Acoustic Edge', 'Phoneme Onset']

# Define all configurations to run
CONFIGURATIONS = [
    # Language comparisons for both participant groups
    {
        'analysis_type': 'language_comparison',
        'participant_group': 'Dutch',
        'stimuli_language': None
    },
    {
        'analysis_type': 'language_comparison',
        'participant_group': 'Chinese',
        'stimuli_language': None
    },
    # Condition comparisons (Words vs Syllables) for Dutch participants
    {
        'analysis_type': 'condition_comparison',
        'participant_group': 'Dutch',
        'stimuli_language': 'Dutch'
    },
    {
        'analysis_type': 'condition_comparison',
        'participant_group': 'Dutch',
        'stimuli_language': 'Chinese'
    },
    # Condition comparisons (Words vs Syllables) for Chinese participants
    {
        'analysis_type': 'condition_comparison',
        'participant_group': 'Chinese',
        'stimuli_language': 'Dutch'
    },
    {
        'analysis_type': 'condition_comparison',
        'participant_group': 'Chinese',
        'stimuli_language': 'Chinese'
    },
]

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def line_graph_final_whole_brain_n_f(x1_all, x2_all, title_all, condition_p_all):
    """
    Visualization for native vs familiar language comparison (Dutch participants).
    Color scheme: familiar=#9e9385, native=#504538
    """
    fig, ax = plt.subplots(2, 1, figsize=(2.8, 3), sharex=True, sharey=True)

    for i in range(2):
        x1 = x1_all[i]
        x2 = x2_all[i]
        condition_p = condition_p_all[i]
        title = title_all[i]

        # Plot first condition (familiar)
        x1_mean = x1.mean('case').x[2:-2]
        time = x1.time.times[2:-2]
        x1_error = [sem(x1.x[:, t]) for t in range(len(time))]

        ax[i].plot(time, x1_mean, '#9e9385', linewidth=0.5)
        ax[i].fill_between(time, x1_mean - x1_error, x1_mean + x1_error,
                          alpha=0.7, color='#9e9385', edgecolor='#9e9385', linewidth=0.2)

        # Plot second condition (native)
        x2_mean = x2.mean('case').x[2:-2]
        x2_error = [sem(x2.x[:, t]) for t in range(len(time))]

        ax[i].plot(time, x2_mean, '#504538', linewidth=0.5)
        ax[i].fill_between(time, x2_mean - x2_error, x2_mean + x2_error,
                          alpha=0.7, color='#504538', edgecolor='#504538', linewidth=0.2)

        # Plot significance markers
        ax[i].plot(time, np.multiply(condition_p, -0.00), 'red', linewidth=2)

        # Formatting
        ax[i].spines['left'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)

        ax[i].yaxis.set_major_locator(MaxNLocator(5))
        ax[i].xaxis.set_major_locator(MaxNLocator(5))

        ax[i].yaxis.grid()
        ax[i].xaxis.grid()

        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_title(title)

    plt.subplots_adjust(hspace=.0)

    fig.text(0.5, 0., 'Time (sec)', ha='center', fontsize=12)
    fig.text(0, 0.5, 'Power of Weights $\mathregular{\sqrt{w^{2}}}$',
            va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.show()


def line_graph_final_whole_brain_n_u(x1_all, x2_all, title_all, condition_p_all):
    """
    Visualization for native vs unfamiliar language comparison (Chinese participants).
    Color scheme: native=#504538, unfamiliar=#e39ba6
    """
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

    for i in range(2):
        x1 = x1_all[i]
        x2 = x2_all[i]
        condition_p = condition_p_all[i]
        title = title_all[i]

        # Plot first condition (native)
        x1_mean = x1.mean('case').x[2:-2]
        time = x1.time.times[2:-2]
        x1_error = [sem(x1.x[:, t]) for t in range(len(time))]

        ax[i].plot(time, x1_mean, '#504538', linewidth=0.5)
        ax[i].fill_between(time, x1_mean - x1_error, x1_mean + x1_error,
                          alpha=0.7, color='#504538', edgecolor='#504538', linewidth=0.2)

        # Plot second condition (unfamiliar)
        x2_mean = x2.mean('case').x[2:-2]
        x2_error = [sem(x2.x[:, t]) for t in range(len(time))]

        ax[i].plot(time, x2_mean, '#e39ba6', linewidth=0.5)
        ax[i].fill_between(time, x2_mean - x2_error, x2_mean + x2_error,
                          alpha=0.7, color='#e39ba6', edgecolor='#e39ba6', linewidth=0.2)

        # Plot significance markers
        ax[i].plot(time, np.multiply(condition_p, -0.00), 'red', linewidth=2)

        # Formatting
        ax[i].spines['left'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)

        ax[i].yaxis.set_major_locator(MaxNLocator(5))
        ax[i].xaxis.set_major_locator(MaxNLocator(5))

        ax[i].yaxis.grid()
        ax[i].xaxis.grid()

        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_title(title)

    plt.subplots_adjust(hspace=.0)

    fig.text(0.5, 0., 'Time (sec)', ha='center', fontsize=12)
    fig.text(0, 0.5, 'Power of Weights $\mathregular{\sqrt{w^{2}}}$',
            va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.show()


def line_graph_final_whole_brain_w_s(x1_all, x2_all, title_all, condition_p_all):
    """
    Visualization for words vs syllables comparison.
    Color scheme: words=#2f4858, syllables=#719e87
    """
    fig, ax = plt.subplots(2, 1,  sharex=True, sharey=True)

    for i in range(2):
        x1 = x1_all[i]
        x2 = x2_all[i]
        condition_p = condition_p_all[i]
        title = title_all[i]

        # Plot first condition (words)
        x1_mean = x1.mean('case').x[2:-2]
        time = x1.time.times[2:-2]
        x1_error = [sem(x1.x[:, t]) for t in range(len(time))]

        ax[i].plot(time, x1_mean, '#2f4858', linewidth=0.5)
        ax[i].fill_between(time, x1_mean - x1_error, x1_mean + x1_error,
                          alpha=0.7, color='#2f4858', edgecolor='#2f4858', linewidth=0.2)

        # Plot second condition (syllables)
        x2_mean = x2.mean('case').x[2:-2]
        x2_error = [sem(x2.x[:, t]) for t in range(len(time))]

        ax[i].plot(time, x2_mean, '#719e87', linewidth=0.5)
        ax[i].fill_between(time, x2_mean - x2_error, x2_mean + x2_error,
                          alpha=0.7, color='#719e87', edgecolor='#719e87', linewidth=0.2)

        # Plot significance markers
        ax[i].plot(time, np.multiply(condition_p, -0.00), 'red', linewidth=2)

        # Formatting
        ax[i].spines['left'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)

        ax[i].yaxis.set_major_locator(MaxNLocator(5))
        ax[i].xaxis.set_major_locator(MaxNLocator(5))

        ax[i].yaxis.grid()
        ax[i].xaxis.grid()

        ax[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[i].set_title(title)

    plt.subplots_adjust(hspace=.0)

    fig.text(0.5, 0., 'Time (sec)', ha='center', fontsize=12)
    fig.text(0, 0.5, 'Power of Weights $\mathregular{\sqrt{w^{2}}}$',
            va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.show()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_subject_info(participant_group):
    """Get subjects and data root for a participant group."""
    
    DATASET = participant_group + '_participants'
    root = Path.cwd().parents[1]
    data_root = root / DATASET / 'processed'
    
    if participant_group == 'Chinese':
        subjects = [
            'sub-021', 'sub-022', 'sub-023', 'sub-024', 'sub-025', 'sub-026',
            'sub-027', 'sub-028', 'sub-029', 'sub-030', 'sub-032', 'sub-033',
            'sub-034', 'sub-035'
        ]
    elif participant_group == 'Dutch':
        subjects = [
            'sub-003', 'sub-005', 'sub-007', 'sub-008', 'sub-009', 'sub-010',
            'sub-012', 'sub-013', 'sub-014', 'sub-015', 'sub-016', 'sub-017',
            'sub-018', 'sub-019', 'sub-020'
        ]
    else:
        raise ValueError("PARTICIPANT_GROUP must be 'Chinese' or 'Dutch'")

    return data_root, subjects


def setup_analysis_parameters(config):
    """Set up analysis parameters based on configuration."""
    analysis_type = config['analysis_type']
    participant_group = config['participant_group']
    stimuli_language = config['stimuli_language']

    if analysis_type == 'language_comparison':
        condition = 'Words'
        test_name = f'Control2_{participant_group}_participants_STG_{condition}'
        model_condition1 = 'Control2_Delta+Theta_STG_sources_normalized_Dutch_stimuli_acoustic+phonemes'
        model_condition2 = 'Control2_Delta+Theta_STG_sources_normalized_Chinese_stimuli_acoustic+phonemes'
        condition1_name = 'Dutch_stimuli'
        condition2_name = 'Chinese_stimuli'
        comparison_type = 'native_vs_unfamiliar'

    elif analysis_type == 'condition_comparison':
        test_name = f'Control2_{participant_group}_participants_{stimuli_language}_stimuli_Words_vs_syllables'
        model = f'Control2_Delta+Theta_STG_sources_normalized_{stimuli_language}_stimuli_acoustic+phonemes'
        model_condition1 = model
        model_condition2 = model
        condition1_name = 'Words'
        condition2_name = 'Syllables'
        comparison_type = 'words_vs_syllables'
        condition = None

    return {
        'test_name': test_name,
        'model_condition1': model_condition1,
        'model_condition2': model_condition2,
        'condition1_name': condition1_name,
        'condition2_name': condition2_name,
        'comparison_type': comparison_type,
        'condition': condition
    }


def load_hemisphere_data(subjects, data_root, model, condition, hemisphere, subject_dir):
    """Load TRF data for a specific hemisphere."""
    rows = []
    x_names = None

    for subject in subjects:
        print(f"  Loading {subject} - {hemisphere}")
        TRF_DIR = data_root / subject / 'meg' / 'TRF' / condition
        trf = eelbrain.load.unpickle(TRF_DIR / f'{subject} {model}_{hemisphere}.pickle')
        trf.r.source._subjects_dir = subject_dir
        rows.append([subject, trf.r, *trf.h])
        x_names = trf.x

    dataset = eelbrain.Dataset.from_caselist(['subject', 'det', *x_names], rows)
    return dataset


def prepare_visualization_data(data_c1_lh, data_c1_rh, data_c2_lh, data_c2_rh,
                               test_name, anova_path):
    """Prepare data for visualization."""
    condition_p = []
    x1 = []
    x2 = []
    title = []

    # Load significance results and prepare data for acoustic edges
    print("  Loading acoustic edge results...")
    acoustic_edge_p = np.load(
        os.path.join(anova_path, f'acoustic_edge_{test_name}_source_whole_brain.npy'),
        allow_pickle=True
    )
    condition_p.append(acoustic_edge_p)

    # Compute acoustic edge weights (averaged across hemispheres and frequency)
    x1.append(
        (data_c1_rh["gammatone_on"].square().sqrt().mean('frequency').mean('source') +
         data_c1_lh["gammatone_on"].square().sqrt().mean('frequency').mean('source')) / 2
    )
    x2.append(
        (data_c2_rh["gammatone_on"].square().sqrt().mean('frequency').mean('source') +
         data_c2_lh["gammatone_on"].square().sqrt().mean('frequency').mean('source')) / 2
    )
    title.append('Acoustic Edges')

    # Load significance results and prepare data for phoneme features
    print("  Loading phoneme onset results...")
    phoneme_onset_p = np.load(
        os.path.join(anova_path, f'phoneme_onset_{test_name}_source_whole_brain.npy'),
        allow_pickle=True
    )
    condition_p.append(phoneme_onset_p)

    # Compute phoneme feature weights
    x1.append(
        (data_c1_rh["phonemes"].square().sqrt().mean('source') +
         data_c1_rh["cohort_surprisal"].square().sqrt().mean('source') +
         data_c1_rh["cohort_entropy"].square().sqrt().mean('source') +
         data_c1_lh["phonemes"].square().sqrt().mean('source') +
         data_c1_lh["cohort_surprisal"].square().sqrt().mean('source') +
         data_c1_lh["cohort_entropy"].square().sqrt().mean('source')) / 6
    )
    x2.append(
        (data_c2_rh["phonemes"].square().sqrt().mean('source') +
         data_c2_rh["cohort_surprisal"].square().sqrt().mean('source') +
         data_c2_rh["cohort_entropy"].square().sqrt().mean('source') +
         data_c2_lh["phonemes"].square().sqrt().mean('source') +
         data_c2_lh["cohort_surprisal"].square().sqrt().mean('source') +
         data_c2_lh["cohort_entropy"].square().sqrt().mean('source')) / 6
    )
    title.append('Phoneme Features')

    return x1, x2, title, condition_p


def generate_visualization(x1, x2, title, condition_p, comparison_type,
                          participant_group, figure_path, config_name):
    """Generate and save visualization."""
    print("  Generating visualization...")

    # Clear any existing figures
    plt.close('all')

    if comparison_type == 'native_vs_unfamiliar':
        if participant_group == 'Dutch':
            line_graph_final_whole_brain_n_f(x1, x2, title, condition_p)
        else:
            line_graph_final_whole_brain_n_u(x2, x1, title, condition_p)
    elif comparison_type == 'words_vs_syllables':
        line_graph_final_whole_brain_w_s(x1, x2, title, condition_p)

    # Save figure
    figure_filename = f'{config_name}_TRF_visualization.svg'
    figure_full_path = os.path.join(figure_path, figure_filename)
    plt.savefig(figure_full_path, dpi=300, bbox_inches='tight', format='svg')
    print(f"  Figure saved: {figure_filename}")


def save_peak_latencies(x1, x2, analysis_type, participant_group, stimuli_language,
                        condition1_name, condition2_name, condition, n_subjects,
                        peak_results_path):
    """Save peak latencies to CSV."""
    print("\n" + "="*80)
    print("SAVING PEAK LATENCIES")
    print("="*80)

    rows = []

    # Define time windows
    time_windows = [
        ('Early_time', 0, 15, -5),
        ('Mid_time', 15, 40, 15),
        ('Late_time', 40, None, 35)
    ]

    if analysis_type == 'language_comparison':
        # Acoustic edges - all time windows
        for time_window_info in time_windows:
            time_window, start, end, offset = time_window_info

            for i in range(n_subjects):
                if end is None:
                    rows.append([condition1_name, condition, time_window, FEATURES[0], i,
                                np.argmax(x1[0][i].x[start:]) + offset])
                else:
                    rows.append([condition1_name, condition, time_window, FEATURES[0], i,
                                np.argmax(x1[0][i].x[start:end]) + offset])
            for i in range(n_subjects):
                if end is None:
                    rows.append([condition2_name, condition, time_window, FEATURES[0], i,
                                np.argmax(x2[0][i].x[start:]) + offset])
                else:
                    rows.append([condition2_name, condition, time_window, FEATURES[0], i,
                                np.argmax(x2[0][i].x[start:end]) + offset])

        # Phoneme features - all time windows
        for time_window_info in time_windows:
            time_window, start, end, offset = time_window_info

            for i in range(n_subjects):
                if end is None:
                    rows.append([condition1_name, condition, time_window, FEATURES[1], i,
                                np.argmax(x1[1][i].x[start:]) + offset])
                else:
                    rows.append([condition1_name, condition, time_window, FEATURES[1], i,
                                np.argmax(x1[1][i].x[start:end]) + offset])
            for i in range(n_subjects):
                if end is None:
                    rows.append([condition2_name, condition, time_window, FEATURES[1], i,
                                np.argmax(x2[1][i].x[start:]) + offset])
                else:
                    rows.append([condition2_name, condition, time_window, FEATURES[1], i,
                                np.argmax(x2[1][i].x[start:end]) + offset])

        output_filename = f'{participant_group}_participants_all_stimuli_{condition}_Delta+Theta_peak_latencies.csv'

    elif analysis_type == 'condition_comparison':
        # Acoustic edges - all time windows
        for time_window_info in time_windows:
            time_window, start, end, offset = time_window_info

            for i in range(n_subjects):
                if end is None:
                    rows.append([condition1_name, stimuli_language, time_window, FEATURES[0], i,
                                np.argmax(x1[0][i].x[start:]) + offset])
                else:
                    rows.append([condition1_name, stimuli_language, time_window, FEATURES[0], i,
                                np.argmax(x1[0][i].x[start:end]) + offset])
            for i in range(n_subjects):
                if end is None:
                    rows.append([condition2_name, stimuli_language, time_window, FEATURES[0], i,
                                np.argmax(x2[0][i].x[start:]) + offset])
                else:
                    rows.append([condition2_name, stimuli_language, time_window, FEATURES[0], i,
                                np.argmax(x2[0][i].x[start:end]) + offset])

        # Phoneme features - all time windows
        for time_window_info in time_windows:
            time_window, start, end, offset = time_window_info

            for i in range(n_subjects):
                if end is None:
                    rows.append([condition1_name, stimuli_language, time_window, FEATURES[1], i,
                                np.argmax(x1[1][i].x[start:]) + offset])
                else:
                    rows.append([condition1_name, stimuli_language, time_window, FEATURES[1], i,
                                np.argmax(x1[1][i].x[start:end]) + offset])
            for i in range(n_subjects):
                if end is None:
                    rows.append([condition2_name, stimuli_language, time_window, FEATURES[1], i,
                                np.argmax(x2[1][i].x[start:]) + offset])
                else:
                    rows.append([condition2_name, stimuli_language, time_window, FEATURES[1], i,
                                np.argmax(x2[1][i].x[start:end]) + offset])

        output_filename = f'{participant_group}_participants_{stimuli_language}_stimuli_Words_vs_Syllables_Delta+Theta_peak_latencies.csv'

    # Create DataFrame and save
    df = pd.DataFrame(data=rows, columns=['Condition', 'Stimuli_or_Condition', 'Time', 'Feature', 'subject', 'peak_latency'])
    output_path = os.path.join(peak_results_path, output_filename)
    df.to_csv(output_path, index=False)

    print(f"\nPeak latencies saved to: {output_path}")
    print(f"Total rows: {len(rows)}")
    print(f"Shape: {df.shape}")

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def run_single_configuration(config):
    """Run analysis for a single configuration."""

    # Output paths
    participant_group = config['participant_group']
    DATASET = participant_group + '_participants'
    root = Path.cwd().parents[1]
    subjects_dir = root / DATASET / 'processed'
    RESULTS_PATH = root / 'Scripts' / 'TRF_weight_analysis' / 'Output'
    PEAK_RESULTS_PATH = root / 'Scripts' / 'Accuracy_analysis' / 'Results_publication'
    FIGURE_OUTPUT_PATH = root / 'Scripts' / 'Visualize_TRF_weights' / 'Output'
    SUBJECT_DIR = root / DATASET / 'processed'

    # Create output directories if they don't exist
    os.makedirs(FIGURE_OUTPUT_PATH, exist_ok=True)
    os.makedirs(PEAK_RESULTS_PATH, exist_ok=True)

    analysis_type = config['analysis_type']
    participant_group = config['participant_group']
    stimuli_language = config['stimuli_language']

    # Create configuration name for file naming
    if analysis_type == 'language_comparison':
        config_name = f'{participant_group}_language_comparison'
    else:
        config_name = f'{participant_group}_{stimuli_language}_words_vs_syllables'

    print("\n" + "="*80)
    print(f"RUNNING CONFIGURATION: {config_name}")
    print("="*80)

    # Get subject information
    data_root, subjects = get_subject_info(participant_group)
    n_subjects = len(subjects)

    # Setup analysis parameters
    params = setup_analysis_parameters(config)

    print(f"Participant Group: {participant_group} ({n_subjects} subjects)")
    print(f"Analysis Type: {analysis_type}")
    if analysis_type == 'condition_comparison':
        print(f"Stimuli Language: {stimuli_language}")
    print(f"Test Name: {params['test_name']}")
    print(f"Data Root: {data_root}")

    # Load data
    print("\nLoading data...")

    if analysis_type == 'language_comparison':
        print(f"\nLoading {params['condition1_name']} data...")
        data_c1_lh = load_hemisphere_data(
            subjects, data_root, params['model_condition1'], params['condition'], 'lh', SUBJECT_DIR
        )
        data_c1_rh = load_hemisphere_data(
            subjects, data_root, params['model_condition1'], params['condition'], 'rh', SUBJECT_DIR
        )

        print(f"\nLoading {params['condition2_name']} data...")
        data_c2_lh = load_hemisphere_data(
            subjects, data_root, params['model_condition2'], params['condition'], 'lh', SUBJECT_DIR
        )
        data_c2_rh = load_hemisphere_data(
            subjects, data_root, params['model_condition2'], params['condition'], 'rh', SUBJECT_DIR
        )

    elif analysis_type == 'condition_comparison':
        print(f"\nLoading {params['condition1_name']} data...")
        data_c1_lh = load_hemisphere_data(
            subjects, data_root, params['model_condition1'], 'Words', 'lh', SUBJECT_DIR
        )
        data_c1_rh = load_hemisphere_data(
            subjects, data_root, params['model_condition1'], 'Words', 'rh', SUBJECT_DIR
        )

        print(f"\nLoading {params['condition2_name']} data...")
        data_c2_lh = load_hemisphere_data(
            subjects, data_root, params['model_condition2'], 'Syllables', 'lh', SUBJECT_DIR
        )
        data_c2_rh = load_hemisphere_data(
            subjects, data_root, params['model_condition2'], 'Syllables', 'rh', SUBJECT_DIR
        )

    print("\nData loading complete.")

    # Prepare visualization data
    print("\nPreparing data for visualization...")
    x1, x2, title, condition_p = prepare_visualization_data(
        data_c1_lh, data_c1_rh, data_c2_lh, data_c2_rh,
        params['test_name'], RESULTS_PATH
    )

    # Generate and save visualization
    generate_visualization(
        x1, x2, title, condition_p, params['comparison_type'],
        participant_group, FIGURE_OUTPUT_PATH, config_name
    )

    # Save peak latencies
    save_peak_latencies(
        x1, x2, analysis_type, participant_group, stimuli_language,
        params['condition1_name'], params['condition2_name'],
        params['condition'], n_subjects, PEAK_RESULTS_PATH
    )

    print(f"\n{'='*80}")
    print(f"CONFIGURATION {config_name} COMPLETE")
    print(f"{'='*80}\n")

# ============================================================================
# RUN ALL CONFIGURATIONS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("BATCH PROCESSING ALL CONFIGURATIONS")
    print("#"*80)
    print(f"\nTotal configurations to process: {len(CONFIGURATIONS)}")
    print("\n" + "#"*80 + "\n")

    successful_configs = 0

    for i, config in enumerate(CONFIGURATIONS, 1):
        print(f"\n{'#'*80}")
        print(f"PROCESSING CONFIGURATION {i}/{len(CONFIGURATIONS)}")
        print(f"{'#'*80}")

        try:
            run_single_configuration(config)
            successful_configs += 1

        except Exception as e:
            print(f"\nERROR processing configuration {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "#"*80)
    print("BATCH PROCESSING COMPLETE")
    print("#"*80)
    print(f"\nSuccessfully processed: {successful_configs}/{len(CONFIGURATIONS)} configurations")
    print("\n" + "#"*80 + "\n")