#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cohort Model: Phoneme Surprisal and Entropy Calculation

This script calculates phoneme surprisal and entropy values for words in stories
based on the cohort model of spoken word recognition. It uses grapheme-to-phoneme
mappings and word frequency counts from a Turkish corpus.

Author: filiztezcan

"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Constants
TURKISH_ALPHABET = [
    'a', 'b', 'c', 'ç', 'd', 'e', 'f', 'g', 'ğ', 'h', 'ı', 'i', 'j', 'k',
    'l', 'm', 'n', 'o', 'ö', 'p', 'r', 's', 'ş', 't', 'u', 'ü', 'v', 'y', 'z'
]

FREQ_FILENAME = 'tur_wikipedia_2021_300K-words_filtered_corrected_app_removed_merged.csv'
OUTPUT_FOLDER_NAME = 'Cohort_model'


class CohortModelAnalyzer:
    """
    Analyzes phoneme sequences using a cohort model to calculate
    surprisal and entropy values.
    """
    
    def __init__(self, dir_path, max_words=100000):
        """
        Initialize the analyzer with directory paths and parameters.
        
        Parameters
        ----------
        dir_path : str
            Current script directory path
        max_words : int
            Maximum number of words to load from frequency file
        """
        self.dir_path = dir_path
        self.max_words = max_words

        
        # Initialize mappings
        self.phone2int = {TURKISH_ALPHABET[i]: i for i in range(len(TURKISH_ALPHABET))}
        self.int2phone = {i: TURKISH_ALPHABET[i] for i in range(len(TURKISH_ALPHABET))}
        self.num_phones = len(self.phone2int)
        
        # Data containers
        self.df_freq = None
        self.df_word_grapheme = None
        self.num_words = 0
        self.num_words_all_cohort = 0
        self.counter = None
        
    def setup_directories(self):
        """Create necessary output directories if they don't exist."""
        output_folder = os.path.join(
            self.dir_path, 'Materials', OUTPUT_FOLDER_NAME
        )
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        return output_folder
    
    def save_phone_mappings(self):
        """Save phoneme-to-integer mappings to pickle files."""
        raw_data_dir = os.path.join(self.dir_path, 'Materials')
        
        with open(os.path.join(raw_data_dir, 'phone2int_cohort.pkl'), 'wb') as f:
            pickle.dump(self.phone2int, f)
            
        with open(os.path.join(raw_data_dir, 'int2phone_cohort.pkl'), 'wb') as f:
            pickle.dump(self.int2phone, f)
    
    def load_frequency_data(self):
        """
        Load and process word frequency data from CSV file.
        
        Returns
        -------
        tuple
            (grapheme_words, phoneme_words) lists
        """
        freq_file_path = os.path.join(
            self.dir_path, 'Materials',  FREQ_FILENAME
        )
        
        print(f"Loading frequency data from: {freq_file_path}")
        self.df_freq = pd.read_csv(freq_file_path, sep=';')
        self.df_freq = self.df_freq[:self.max_words]
        
        # Convert words to phoneme sequences (graphemes in this case)
        words = self.df_freq['Word'].tolist()
        phonemes = [self._clean_word(word) for word in words]
        
        # Create dataframe with grapheme and phoneme representations
        self.df_word_grapheme = pd.DataFrame({
            'grapheme': words,
            'phoneme': phonemes
        })
        
        self.num_words = len(self.df_freq)
        self.num_words_all_cohort = self.df_freq['Freq'].sum()
        
        print(f"Loaded {self.num_words} words with total frequency: {self.num_words_all_cohort}")
        
        return words, phonemes
    
    @staticmethod
    def _clean_word(word):
        """
        Remove non-alphabetic characters from word.
        
        Parameters
        ----------
        word : str
            Input word
            
        Returns
        -------
        list
            List of alphabetic characters
        """
        return [char for char in word if char.isalpha()]
    
    def calculate_initial_phoneme_frequencies(self):
        """
        Calculate frequency of each initial phoneme across all words in cohort.
        
        Returns
        -------
        np.ndarray
            Counter array with phoneme frequencies
        """
        # Initialize with uniform prior
        self.counter = np.ones(self.num_phones) / self.num_words_all_cohort
        
        # Add frequency-weighted counts for initial phonemes
        for idx in range(self.num_words):
            freq = float(self.df_freq['Freq'].iloc[idx])
            first_phoneme = self.df_word_grapheme['phoneme'].iloc[idx][0]
            phoneme_idx = self.phone2int[first_phoneme]
            self.counter[phoneme_idx] += freq / self.num_words_all_cohort
        
        print("Calculated initial phoneme frequencies")
        return self.counter
    
    def load_story_phonemes(self, phoneme_folder):
        """
        Load phoneme transcription files for stories.
        
        Parameters
        ----------
        phoneme_folder : str
            Path to folder containing phoneme transcription files
            
        Returns
        -------
        list
            List of phoneme transcription filenames
        """
        phoneme_files = [
            f for f in os.listdir(phoneme_folder)
            if f.endswith('.txt')
        ]
        print(f"Found {len(phoneme_files)} phoneme transcription files")
        return phoneme_files
    
    def parse_story_words(self, df_phonemes):
        """
        Parse phoneme dataframe to extract word-level phoneme sequences.
        
        Parameters
        ----------
        df_phonemes : pd.DataFrame
            Dataframe with phoneme-level transcriptions
            
        Returns
        -------
        tuple
            (phonemes_words, grapheme_words) lists of word sequences
        """
        phonemes_words = []
        grapheme_words = []
        
        j = 0
        while j < len(df_phonemes):
            phoneme_word = []
            current_word = df_phonemes['words'].iloc[j]
            
            # Collect all phonemes for the current word
            while j < len(df_phonemes) and df_phonemes['words'].iloc[j] == current_word:
                phoneme_word.append(df_phonemes['phonemes'].iloc[j])
                j += 1
            
            phonemes_words.append(phoneme_word)
            grapheme_words.append(current_word)
        
        return phonemes_words, grapheme_words
    
    def calculate_cohort_metrics(self, test_word_phonemes):
        """
        Calculate entropy and surprisal for each phoneme in a test word
        using the cohort model.
        
        Parameters
        ----------
        test_word_phonemes : list
            List of phonemes in the test word
            
        Returns
        -------
        tuple
            (shannon_entropy, surprisal) arrays for each phoneme
        """
        if test_word_phonemes == 'UNK':
            return None, None
        
        word_length = len(test_word_phonemes)
        
        # Initialize probability matrix
        prob_matrix = np.ones((word_length, self.num_phones)) / self.num_words_all_cohort
        prob_matrix[0, :] = self.counter  # Initial phoneme probabilities
        
        # Build cohort incrementally
        cohort = self.df_word_grapheme['phoneme'].tolist()
        
        for k in range(word_length - 1):
            # Filter cohort to words matching up to position k
            cohort = [
                word for word in cohort
                if len(word) > k and word[k] == test_word_phonemes[k]
            ]
            
            # Calculate probabilities for next phoneme position
            for word in cohort:
                if len(word) > k + 1:
                    grapheme = ''.join(word)
                    freq = self._get_word_frequency(grapheme)
                    
                    next_phoneme = word[k + 1]
                    phoneme_idx = self.phone2int[next_phoneme]
                    prob_matrix[k + 1, phoneme_idx] += freq / self.num_words_all_cohort
        
        # Calculate entropy and surprisal
        shannon = np.zeros(word_length)
        surprisal = np.zeros(word_length)
        
        for k in range(word_length):
            # Normalize probabilities
            prob_sum = prob_matrix[k].sum()
            normalized_probs = prob_matrix[k] / prob_sum
            
            # Shannon entropy: H = -sum(p * log2(p))
            with np.errstate(divide='ignore', invalid='ignore'):
                shannon[k] = -np.sum(
                    normalized_probs * np.log2(normalized_probs + 1e-10)
                )
            
            # Surprisal: -log2(p(phoneme))
            phoneme_idx = self.phone2int[test_word_phonemes[k]]
            surprisal[k] = -np.log2(
                prob_matrix[k, phoneme_idx] / prob_sum + 1e-10
            )
        
        return shannon, surprisal
    
    def _get_word_frequency(self, grapheme):
        """
        Get frequency count for a word from the frequency dataframe.
        
        Parameters
        ----------
        grapheme : str
            Word in grapheme form
            
        Returns
        -------
        float
            Frequency count for the word
        """
        match = self.df_freq.loc[self.df_freq['Word'] == grapheme]
        
        if len(match) == 0:
            return 1 / self.num_words_all_cohort  # Minimum frequency
        else:
            return float(match.iloc[-1]['Freq'])
    
    def process_story_file(self, phoneme_file, phoneme_folder, output_folder):
        """
        Process a single story file to calculate cohort metrics.
        
        Parameters
        ----------
        phoneme_file : str
            Filename of phoneme transcription
        phoneme_folder : str
            Path to phoneme folder
        output_folder : str
            Path to output folder
        """
        print(f"\nProcessing: {phoneme_file}")
        
        # Load phoneme data
        df_phonemes = pd.read_table(
            os.path.join(phoneme_folder, phoneme_file),
            encoding='utf-8',
            sep=','
        )
        
        # Parse word-level phoneme sequences
        phonemes_words_test, _ = self.parse_story_words(df_phonemes)
        
        # Calculate metrics for each word
        shannon_all = []
        surprisal_all = []
        
        for i, word_phonemes in enumerate(phonemes_words_test):
            if i % 50 == 0:
                print(f"  Processing word {i}/{len(phonemes_words_test)}")
            
            shannon, surprisal = self.calculate_cohort_metrics(word_phonemes)
            
            if shannon is not None:
                shannon_all.extend(shannon)
                surprisal_all.extend(surprisal)
        
        # Create output dataframe
        df_cohort = pd.DataFrame({
            'cohort_entropy': shannon_all,
            'cohort_surprisal': surprisal_all
        })
        
        df_all = pd.concat([df_phonemes, df_cohort], axis=1)
        
        # Save results
        output_filename = phoneme_file.replace('.txt', '_cohort_model.csv')
        output_path = os.path.join(output_folder, output_filename)
        
        df_all.to_csv(output_path, index=False, sep=';', line_terminator='\n')
        print(f"  Saved results to: {output_filename}")
    
    def run_analysis(self, max_files=None):
        """
        Run the complete cohort model analysis pipeline.
        
        Parameters
        ----------
        max_files : int, optional
            Maximum number of story files to process (for testing)
        """
        print("=" * 70)
        print("Starting Cohort Model Analysis")
        print("=" * 70)
        
        # Setup
        output_folder = self.setup_directories()
        self.save_phone_mappings()
        
        # Load frequency data
        self.load_frequency_data()
        self.calculate_initial_phoneme_frequencies()
        
        # Load story files
        phoneme_folder = os.path.join(
            self.dir_path, 'Materials', 'Stimuli', 'Transcription_full_instances'
        )
        phoneme_files = self.load_story_phonemes(phoneme_folder)
        
        # Limit number of files if specified
        if max_files is not None:
            phoneme_files = phoneme_files[:max_files]
            print(f"Processing first {max_files} files only")
        
        # Process each story file
        for idx, phoneme_file in enumerate(phoneme_files):
            print(f"\n[{idx + 1}/{len(phoneme_files)}] ", end="")
            self.process_story_file(phoneme_file, phoneme_folder, output_folder)
        
        print("\n" + "=" * 70)
        print("Analysis complete!")
        print("=" * 70)


def main():
    """Main execution function."""
    dir_path = Path.cwd().parents[0]
    
    # Initialize analyzer
    analyzer = CohortModelAnalyzer(dir_path, max_words=100000)
    
    # Run analysis (process first 20 files as in original script)
    analyzer.run_analysis(max_files=20)


if __name__ == '__main__':
    main()
