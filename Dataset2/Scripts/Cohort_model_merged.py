#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cohort Model: Phoneme Surprisal and Entropy Calculation

This script calculates phoneme surprisal and entropy values for words/syllables in stories
based on a cohort model, using grapheme-to-phoneme dictionary and frequency data.

Supports both Dutch and Chinese datasets.

Author: filiztezcan
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


class PronunciationDictionary:
    """Manages phoneme vocabulary and mappings."""
    
    def __init__(self, sort_phones=False):
        """
        Initialize pronunciation dictionary.
        
        Args:
            sort_phones: If True, sort phonemes for consistent mapping (used for Chinese)
        """
        self.phone2int = {}
        self.int2phone = {}
        self.phoneme_word_list = []
        self.all_phones = []
        self.sort_phones = sort_phones
        
    def build_from_text(self, text_data):
        """Build phoneme mappings from grapheme-to-phoneme text data.
        
        Args:
            text_data: List of [grapheme, phoneme] pairs
        """
        for grapheme, phonemes_str in text_data:
            # Clean and split phonemes
            phonemes = [p for p in phonemes_str.split(' ') if p]
            self.all_phones.extend(phonemes)
            self.phoneme_word_list.append(' '.join(phonemes))
        
        # Create bidirectional mapping
        if self.sort_phones:
            unique_phones = sorted(set(self.all_phones))
        else:
            unique_phones = list(set(self.all_phones))
            
        for i, phone in enumerate(unique_phones):
            self.phone2int[phone] = i
            self.int2phone[i] = phone


class CohortModelConfig:
    """Configuration for cohort model processing."""
    
    # Supported languages
    DUTCH = 'Dutch'
    CHINESE = 'Chinese'
    
    # Processing units
    WORDS = 'Words'
    SYLLABLES = 'Syllables'
    
    # Participant groups
    DUTCH_PARTICIPANTS = 'Dutch_participants'
    CHINESE_PARTICIPANTS = 'Chinese_participants'
    
    # Stimuli types
    DUTCH_STIMULI = 'Dutch_stimuli'
    CHINESE_STIMULI = 'Chinese_stimuli'
    
    def __init__(self, language, condition, dataset, stimuli):
        """
        Initialize configuration.
        
        Args:
            language: 'Dutch' or 'Chinese'
            condition: 'Words' or 'Syllables'
            dataset: 'Dutch_participants' or 'Chinese_participants'
            stimuli: 'Dutch_stimuli' or 'Chinese_stimuli'
        """
        self.language = language
        self.condition = condition
        self.dataset = dataset
        self.stimuli = stimuli
        
        # Set language-specific parameters
        if language == self.DUTCH:
            self.dict_filename = 'Dutch_dict_2022.txt'
            self.freq_filename = 'SUBTLEX-NL_filtered_2022_cut.csv'
            self.freq_column = 'SUBTLEXWF'
            self.num_words_all_cohort = 970843
            self.freq_file_sep = ';'
            self.freq_reader = pd.read_csv
            self.phoneme_file_ext = '.csv'
            self.batch_size = 20
            self.sort_phones = False
            self.save_dict_pickle = True
        elif language == self.CHINESE:
            self.dict_filename = 'Chinese_dict.txt'
            self.freq_filename = 'Chinese_freq_file.xlsx'
            self.freq_column = 'W_million'
            self.num_words_all_cohort = 999016
            self.freq_file_sep = None
            self.freq_reader = pd.read_excel
            self.phoneme_file_ext = '.txt'
            self.batch_size = 70
            self.sort_phones = True
            self.save_dict_pickle = False
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def get_paths(self, root_path):
        """Get file paths based on configuration.
        
        Args:
            root_path: Root directory path
            
        Returns:
            Dictionary with all necessary paths
        """
        dict_folder = root_path / 'Materials' / self.stimuli
        
        paths = {
            'dict_folder': dict_folder,
            'dict_path': dict_folder / self.dict_filename,
            'freq_path': dict_folder / self.freq_filename,
            'phoneme_folder': dict_folder / 'Stimuli' / 'Transription' / self.dataset / self.condition,
            'output_folder': dict_folder / f'Cohort_model_{self.dataset}' / self.condition
        }
        
        return paths


def load_dictionary(filepath):
    """Load grapheme-to-phoneme dictionary from file.
    
    Args:
        filepath: Path to dictionary file (tab-separated)
        
    Returns:
        List of [grapheme, phoneme] pairs
    """
    all_words = []
    with open(filepath, encoding='utf8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                all_words.append([parts[0], parts[1]])
    return all_words


def calculate_initial_phoneme_frequencies(dictionary, df_freq, df_word_grapheme, 
                                          num_words_all_cohort, freq_column):
    """Calculate frequency of initial phonemes across the corpus.
    
    Args:
        dictionary: PronunciationDictionary object
        df_freq: DataFrame with word frequencies
        df_word_grapheme: DataFrame mapping graphemes to phonemes
        num_words_all_cohort: Total word count in corpus
        freq_column: Name of frequency column in df_freq
        
    Returns:
        Tuple of (initial_counter, phoneme_words_all, graph_words_all)
    """
    num_phones = len(dictionary.phone2int)
    counter = np.ones(num_phones) / num_words_all_cohort
    
    phoneme_words_all = []
    graph_words_all = []
    missing_count = 0
    
    for _, row in df_freq.iterrows():
        word = row['Word']
        if word in df_word_grapheme['grapheme'].values:
            graph_words_all.append(word)
            freq = float(row[freq_column])
            
            # Get phoneme transcription
            index = df_word_grapheme.loc[df_word_grapheme['grapheme'] == word].index[0]
            phoneme_word = df_word_grapheme['phoneme'][index]
            phoneme_list = phoneme_word.split(' ')
            phoneme_words_all.append(phoneme_list)
            
            # Update initial phoneme counter
            initial_phone = phoneme_list[0]
            counter[dictionary.phone2int[initial_phone]] += freq / num_words_all_cohort
        else:
            missing_count += 1
    
    if missing_count > 0:
        print(f"Total words without phoneme transcription: {missing_count}")
    
    return counter, phoneme_words_all, graph_words_all


def extract_word_phonemes(df_phonemes):
    """Extract word-level phoneme sequences from phoneme-level data.
    
    Args:
        df_phonemes: DataFrame with 'words' and 'phonemes' columns
        
    Returns:
        Tuple of (phoneme_sequences, grapheme_words)
    """
    phonemes_words = []
    grapheme_words = []
    
    j = 0
    while j < len(df_phonemes):
        phoneme_word = []
        current_word = df_phonemes['words'][j]
        
        # Collect all phonemes for the current word
        while j < len(df_phonemes) and df_phonemes['words'][j] == current_word:
            phoneme_word.append(df_phonemes['phonemes'][j])
            j += 1
            
        phonemes_words.append(phoneme_word)
        grapheme_words.append(current_word)
    
    return phonemes_words, grapheme_words


def calculate_cohort_probabilities(test_phonemes, phoneme_words_all, dictionary,
                                   df_word_grapheme, df_freq, num_words_all_cohort,
                                   initial_counter, freq_column):
    """Calculate phoneme-by-phoneme probabilities using cohort model.
    
    Args:
        test_phonemes: List of phonemes in test word
        phoneme_words_all: List of all phoneme sequences in corpus
        dictionary: PronunciationDictionary object
        df_word_grapheme: DataFrame mapping graphemes to phonemes
        df_freq: DataFrame with word frequencies
        num_words_all_cohort: Total word count in corpus
        initial_counter: Initial phoneme probabilities
        freq_column: Name of frequency column in df_freq
        
    Returns:
        Probability matrix (n_phonemes x n_phones)
    """
    num_phones = len(dictionary.phone2int)
    prob_matrix = np.ones((len(test_phonemes), num_phones)) / num_words_all_cohort
    prob_matrix[0, :] = initial_counter
    
    cohort = phoneme_words_all[:]
    
    # Build probability matrix phoneme by phoneme
    for k in range(len(test_phonemes) - 1):
        # Filter cohort: keep words matching up to position k
        cohort = [word for word in cohort 
                 if len(word) > k and word[k] == test_phonemes[k]]
        
        # Update probabilities for next position
        for word in cohort:
            if len(word) <= k + 1:
                continue
                
            # Get word frequency
            phoneme_str = ' '.join(word)
            if phoneme_str not in df_word_grapheme['phoneme'].values:
                freq = 1 / num_words_all_cohort
            else:
                index = df_word_grapheme.loc[df_word_grapheme['phoneme'] == phoneme_str].index[0]
                grapheme = df_word_grapheme['grapheme'][index]
                freq_row = df_freq.loc[df_freq['Word'] == grapheme]
                
                if len(freq_row) == 0:
                    freq = 1 / num_words_all_cohort
                else:
                    freq = float(freq_row.iloc[-1][freq_column])
            
            # Update probability for next phoneme
            next_phone = word[k + 1]
            prob_matrix[k + 1, dictionary.phone2int[next_phone]] += freq / num_words_all_cohort
    
    return prob_matrix


def calculate_information_measures(prob_matrix, test_phonemes, dictionary):
    """Calculate entropy and surprisal for each phoneme.
    
    Args:
        prob_matrix: Probability matrix from cohort model
        test_phonemes: List of phonemes
        dictionary: PronunciationDictionary object
        
    Returns:
        Tuple of (entropy_values, surprisal_values)
    """
    entropy = np.zeros(len(test_phonemes))
    surprisal = np.zeros(len(test_phonemes))
    
    for k in range(len(test_phonemes)):
        # Entropy: -sum(p * log2(p))
        prob_k = prob_matrix[k]
        prob_k = prob_k[prob_k > 0]  # Avoid log(0)
        entropy[k] = -np.sum(prob_k * np.log2(prob_k))
        
        # Surprisal: -log2(p(observed))
        phone_idx = dictionary.phone2int[test_phonemes[k]]
        prob_observed = prob_matrix[k, phone_idx] / prob_matrix[k].sum()
        surprisal[k] = -np.log2(prob_observed) if prob_observed > 0 else np.inf
    
    return entropy, surprisal


def get_word_frequency_surprisal(grapheme, df_freq, num_words_all_cohort, freq_column):
    """Get word-level frequency surprisal.
    
    Args:
        grapheme: Word in graphemic form
        df_freq: DataFrame with word frequencies
        num_words_all_cohort: Total word count in corpus
        freq_column: Name of frequency column in df_freq
        
    Returns:
        Word frequency surprisal or 'NAN' if not found
    """
    freq_row = df_freq.loc[df_freq['Word'] == grapheme.lower()]
    if len(freq_row) > 0:
        word_freq = float(freq_row.iloc[-1][freq_column])
        return -np.log2(word_freq / num_words_all_cohort)
    else:
        print(f"Word frequency not found: {grapheme}")
        return 'NAN'


def process_story_file(phoneme_file, phoneme_folder, phoneme_words_all, dictionary,
                      df_word_grapheme, df_freq, config, initial_counter,
                      output_folder, verbose=False):
    """Process a single story file and calculate cohort model values.
    
    Args:
        phoneme_file: Name of the phoneme file
        phoneme_folder: Path to folder containing phoneme files
        phoneme_words_all: List of all phoneme sequences in corpus
        dictionary: PronunciationDictionary object
        df_word_grapheme: DataFrame mapping graphemes to phonemes
        df_freq: DataFrame with word frequencies
        config: CohortModelConfig object
        initial_counter: Initial phoneme probabilities
        output_folder: Path to output folder
        verbose: If True, print progress for each word
    """
    print(f"Processing: {phoneme_file}")
    
    # Load phoneme data
    df_phonemes = pd.read_csv(
        os.path.join(phoneme_folder, phoneme_file),
        encoding='utf-8',
        sep=','
    )
    
    # Extract word-level phoneme sequences
    phonemes_words_test, grapheme_words_test = extract_word_phonemes(df_phonemes)
    
    # Calculate cohort model values for each word
    entropy_all = []
    surprisal_all = []
    word_freq_all = []
    
    for i, test_phonemes in enumerate(phonemes_words_test):
        if verbose:
            print(f"  Word {i + 1}/{len(phonemes_words_test)}")
        
        if test_phonemes == ['UNK']:
            continue
        
        # Calculate probabilities using cohort model
        prob_matrix = calculate_cohort_probabilities(
            test_phonemes, phoneme_words_all, dictionary,
            df_word_grapheme, df_freq, config.num_words_all_cohort, 
            initial_counter, config.freq_column
        )
        
        # Calculate entropy and surprisal
        entropy, surprisal = calculate_information_measures(
            prob_matrix, test_phonemes, dictionary
        )
        
        # Get word frequency
        word_freq = get_word_frequency_surprisal(
            grapheme_words_test[i], df_freq, config.num_words_all_cohort, 
            config.freq_column
        )
        
        # Store results (one value per phoneme)
        entropy_all.extend(entropy)
        surprisal_all.extend(surprisal)
        word_freq_all.extend([word_freq] * len(test_phonemes))
    
    # Combine with original data
    df_cohort = pd.DataFrame({
        'cohort_entropy': entropy_all,
        'cohort_surprisal': surprisal_all,
        'word_freq': word_freq_all
    })
    df_all = pd.concat([df_phonemes, df_cohort], axis=1)
    
    # Save results
    output_filename = phoneme_file.replace(config.phoneme_file_ext, '_cohort_model.csv')
    output_path = os.path.join(output_folder, output_filename)
    df_all.to_csv(output_path, index=False, sep=';', line_terminator='\n')
    print(f"Saved: {output_filename}\n")


def main():
    """Main processing pipeline."""
    
    # ==================== CONFIGURATION ====================
    # Set these parameters according to your dataset
    
    LANGUAGE = CohortModelConfig.DUTCH  # or CohortModelConfig.CHINESE
    CONDITION = CohortModelConfig.WORDS  # or CohortModelConfig.SYLLABLES
    DATASET = CohortModelConfig.DUTCH_PARTICIPANTS  # or CohortModelConfig.CHINESE_PARTICIPANTS
    STIMULI = CohortModelConfig.DUTCH_STIMULI  # or CohortModelConfig.CHINESE_STIMULI
    
    # Optional: Set to 'all' to process all files, or specify batch number (1, 2, 3, ...)
    FILE_BATCH = 1
    
    # Optional: Set to True for detailed progress output (useful for Chinese with many words)
    VERBOSE = False
    
    # =======================================================
    
    # Initialize configuration
    config = CohortModelConfig(LANGUAGE, CONDITION, DATASET, STIMULI)
    
    # Get paths
    root = Path.cwd().parents[0]
    paths = config.get_paths(root)
    
    # Create output folder
    os.makedirs(paths['output_folder'], exist_ok=True)
    
    # Load data
    print("Loading dictionary and frequency data...")
    all_words = load_dictionary(paths['dict_path'])
    
    if config.freq_file_sep:
        df_freq = pd.read_csv(paths['freq_path'], sep=config.freq_file_sep)
    else:
        df_freq = config.freq_reader(paths['freq_path'])
    
    # Build pronunciation dictionary
    print("Building pronunciation dictionary...")
    dictionary = PronunciationDictionary(sort_phones=config.sort_phones)
    dictionary.build_from_text(all_words)
    
    # Save dictionary mappings (for Dutch only)
    if config.save_dict_pickle:
        with open(os.path.join(paths['dict_folder'], 'phone2int_cohort.pkl'), "wb") as f:
            pickle.dump(dictionary.phone2int, f)
        with open(os.path.join(paths['dict_folder'], 'int2phone_cohort.pkl'), "wb") as f:
            pickle.dump(dictionary.int2phone, f)
    
    # Create grapheme-phoneme mapping dataframe
    df_word_grapheme = pd.DataFrame(all_words, columns=['grapheme', 'phoneme'])
    
    # Calculate initial phoneme frequencies
    print("Calculating initial phoneme frequencies...")
    initial_counter, phoneme_words_all, _ = calculate_initial_phoneme_frequencies(
        dictionary, df_freq, df_word_grapheme, config.num_words_all_cohort, 
        config.freq_column
    )
    
    # Get story files
    phoneme_files = sorted([f for f in os.listdir(paths['phoneme_folder']) 
                           if f.endswith(config.phoneme_file_ext)])
    
    # Select batch if specified
    if FILE_BATCH != 'all':
        phoneme_files = phoneme_files[(FILE_BATCH - 1) * config.batch_size : 
                                     FILE_BATCH * config.batch_size]
    
    print(f"\nProcessing {len(phoneme_files)} story files...")
    print(f"Language: {LANGUAGE}, Condition: {CONDITION}, Dataset: {DATASET}\n")
    
    # Process each story file
    for phoneme_file in phoneme_files:
        output_filename = phoneme_file.replace(config.phoneme_file_ext, '_cohort_model.csv')
        output_path = os.path.join(paths['output_folder'], output_filename)
        
        # Skip if already processed (for Dutch only)
        if LANGUAGE == CohortModelConfig.DUTCH and os.path.isfile(output_path):
            print(f"Skipping (already exists): {phoneme_file}")
            continue
        
        process_story_file(
            phoneme_file, paths['phoneme_folder'], phoneme_words_all, dictionary,
            df_word_grapheme, df_freq, config, initial_counter,
            paths['output_folder'], verbose=VERBOSE
        )
    
    print("Processing complete!")


if __name__ == "__main__":
    main()