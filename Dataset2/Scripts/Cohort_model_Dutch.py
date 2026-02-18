#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cohort Model: Phoneme Surprisal and Entropy Calculation

This script calculates phoneme surprisal and entropy values for words in stories
based on a cohort model, using grapheme-to-phoneme dictionary and frequency data.

Author: filiztezcan

"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

class PronunciationDictionary:
    """Manages phoneme vocabulary and mappings."""
    
    def __init__(self):
        self.phone2int = {}
        self.int2phone = {}
        self.phoneme_word_list = []
        self.all_phones = []
        
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
        unique_phones = list(set(self.all_phones))
        for i, phone in enumerate(unique_phones):
            self.phone2int[phone] = i
            self.int2phone[i] = phone


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
                                          num_words_all_cohort):
    """Calculate frequency of initial phonemes across the corpus.
    
    Args:
        dictionary: PronunciationDictionary object
        df_freq: DataFrame with word frequencies
        df_word_grapheme: DataFrame mapping graphemes to phonemes
        num_words_all_cohort: Total word count in corpus
        
    Returns:
        numpy array of initial phoneme probabilities
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
            freq = float(row['SUBTLEXWF'])
            
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
            print(f"Missing transcription: {word}")
    
    print(f"\nTotal words without phoneme transcription: {missing_count}")
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
                                   initial_counter):
    """Calculate phoneme-by-phoneme probabilities using cohort model.
    
    Args:
        test_phonemes: List of phonemes in test word
        phoneme_words_all: List of all phoneme sequences in corpus
        dictionary: PronunciationDictionary object
        df_word_grapheme: DataFrame mapping graphemes to phonemes
        df_freq: DataFrame with word frequencies
        num_words_all_cohort: Total word count in corpus
        initial_counter: Initial phoneme probabilities
        
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
                freq = float(freq_row.iloc[-1]['SUBTLEXWF']) if len(freq_row) > 0 else 1 / num_words_all_cohort
            
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


def get_word_frequency_surprisal(grapheme, df_freq, num_words_all_cohort):
    """Get word-level frequency surprisal.
    
    Args:
        grapheme: Word in graphemic form
        df_freq: DataFrame with word frequencies
        num_words_all_cohort: Total word count in corpus
        
    Returns:
        Word frequency surprisal or 'NAN' if not found
    """
    freq_row = df_freq.loc[df_freq['Word'] == grapheme.lower()]
    if len(freq_row) > 0:
        word_freq = float(freq_row.iloc[-1]['SUBTLEXWF'])
        return -np.log2(word_freq / num_words_all_cohort)
    else:
        print(f"Word frequency not found: {grapheme}")
        return 'NAN'


def main():
    """Main processing pipeline."""
    
    # Choose dataset: 'Words' or 'Syllables'
    Condition = 'Words'
    
    # Choose dataset: 'Dutch_participants' or 'Chinese_participants'
    DATASET = 'Dutch_participants'
    
    # Choose stimuli: 'Dutch_stimuli' or 'Chinese_stimuli'
    stimuli = 'Dutch_stimuli'
    
    # Paths
    root = Path.cwd().parents[0]
    dict_folder = root / 'Materials' / stimuli
       
    dict_filename = 'Dutch_dict_2022.txt'
    freq_filename = 'SUBTLEX-NL_filtered_2022_cut.csv'
    
    dict_path = os.path.join(dict_folder, dict_filename)
    freq_file_path = os.path.join(dict_folder, freq_filename)
    
    # Configuration

    num_words_all_cohort = 970843  # Total words in corpus


    phoneme_folder = dict_folder / 'Stimuli' / 'Transription' / DATASET / Condition
    
    
    Cohort_folder = 'Cohort_model_' + DATASET
    output_folder = dict_folder / Cohort_folder / Condition
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Load data
    print("Loading dictionary and frequency data...")
    all_words = load_dictionary(dict_path)
    df_freq = pd.read_csv(freq_file_path, sep=';')
    
    # Build pronunciation dictionary
    print("Building pronunciation dictionary...")
    dictionary = PronunciationDictionary()
    dictionary.build_from_text(all_words)
    
    # Save dictionary mappings
    with open(os.path.join(dict_folder, 'phone2int_cohort.pkl'), "wb") as f:
        pickle.dump(dictionary.phone2int, f)
    with open(os.path.join(dict_folder, 'int2phone_cohort.pkl'), "wb") as f:
        pickle.dump(dictionary.int2phone, f)
    
    # Create grapheme-phoneme mapping dataframe
    df_word_grapheme = pd.DataFrame(all_words, columns=['grapheme', 'phoneme'])
    
    # Calculate initial phoneme frequencies
    print("Calculating initial phoneme frequencies...")
    initial_counter, phoneme_words_all, _ = calculate_initial_phoneme_frequencies(
        dictionary, df_freq, df_word_grapheme, num_words_all_cohort
    )
    
    # Process story files
    phoneme_files = sorted([f for f in os.listdir(phoneme_folder) if f.endswith('.csv')])
    file_batch = 1
    phoneme_files = phoneme_files[(file_batch - 1) * 20 : file_batch * 20]
    
    print(f"\nProcessing {len(phoneme_files)} story files...")
    
    for phoneme_file in phoneme_files:
        output_filename = phoneme_file.replace('.csv', '_cohort_model.csv')
        output_path = os.path.join(output_folder, output_filename)
        
        if os.path.isfile(output_path):
            print(f"Skipping (already exists): {phoneme_file}")
            continue
            
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
            if test_phonemes == ['UNK']:
                continue
            
            # Calculate probabilities using cohort model
            prob_matrix = calculate_cohort_probabilities(
                test_phonemes, phoneme_words_all, dictionary,
                df_word_grapheme, df_freq, num_words_all_cohort, initial_counter
            )
            
            # Calculate entropy and surprisal
            entropy, surprisal = calculate_information_measures(
                prob_matrix, test_phonemes, dictionary
            )
            
            # Get word frequency
            word_freq = get_word_frequency_surprisal(
                grapheme_words_test[i], df_freq, num_words_all_cohort
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
        df_all.to_csv(output_path, index=False, sep=';', line_terminator='\n')
        print(f"Saved: {output_filename}")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()