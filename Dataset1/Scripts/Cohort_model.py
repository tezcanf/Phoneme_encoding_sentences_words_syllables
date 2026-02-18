# -*- coding: utf-8 -*-
"""
Cohort Model for Phoneme Surprisal and Entropy Calculation

This script calculates phoneme surprisal and entropy values for words in stories
based on the cohort model, using:
- Grapheme-to-phoneme dictionary file
- Word frequency count file

@author: filiztezcan
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def load_text_data(filename, language):
    """
    Reads the grapheme-to-phoneme dictionary file.
    
    Parameters
    ----------
    filename : str
        Path to the dictionary file
    language : str
        Language of the dictionary ('French' or other)
        
    Returns
    -------
    list
        List of [grapheme, phoneme] pairs
    """
    all_words = []
    
    if language == 'French':
        df = pd.read_csv(filename, sep=';', encoding='latin1')
        for i in range(len(df)):
            words = [df['grapheme'][i], df['phoneme'][i]]
            all_words.append(words)
    else:
        with open(filename, encoding='utf8') as reader:
            for line in reader:
                line = line.split('\t')
                line[-1] = line[-1].rstrip('\n')  # Remove newline character
                if len(line) > 0:
                    all_words.append(line)
    
    return all_words


class PrononVocab:
    """Class to manage phoneme vocabulary and mappings."""
    
    def __init__(self):
        self.phone2int = {}
        self.int2phone = {}
        self.phoneme_word_list = []
        self.all_phones = []
    
    def __call__(self, text):
        """
        Process text to extract all phonemes and create phoneme word list.
        
        Parameters
        ----------
        text : list
            List of [grapheme, phoneme] pairs
        """
        for word_pair in text:
            phonemes = str(word_pair[1]).split(' ')
            # Remove empty strings
            phonemes = [ph for ph in phonemes if ph != '']
            
            self.all_phones.extend(phonemes)
            phonemes_clean = ' '.join(phonemes)
            self.phoneme_word_list.append(phonemes_clean)


def calculate_cohort_metrics(phonemes_words_test, grapheme_words_test, 
                            phoneme_words_all, df_word_grapheme, df_freq,
                            num_words_all_cohort, phone2int, Counter, num_phones):
    """
    Calculate entropy and surprisal metrics for words based on cohort model.
    
    Parameters
    ----------
    phonemes_words_test : list
        List of phoneme sequences for test words
    grapheme_words_test : list
        List of graphemes for test words
    phoneme_words_all : list
        All phoneme transcriptions from frequency file
    df_word_grapheme : DataFrame
        Mapping between graphemes and phonemes
    df_freq : DataFrame
        Word frequency data
    num_words_all_cohort : int
        Total number of words in the cohort
    phone2int : dict
        Phoneme to integer mapping
    Counter : np.array
        Initial phoneme probabilities
    num_phones : int
        Number of unique phonemes
        
    Returns
    -------
    tuple
        (shannon_all, surprisal_all, word_freq_in_story_all)
    """
    shannon_all = []
    surprisal_all = []
    word_freq_in_story_all = []
    
    for i, phoneme_word in enumerate(phonemes_words_test):
        if phoneme_word == 'UNK':
            continue
            
        phoneme_words_all_temp = phoneme_words_all[:]
        
        # Initialize probability matrix
        Prob_matrix = np.ones((len(phoneme_word), num_phones)) / num_words_all_cohort
        Prob_matrix[0, :] = Counter
        
        # Calculate probabilities for each phoneme position
        for k in range(len(phoneme_word) - 1):
            # Find remaining words in cohort matching up to position k
            res = []
            for j, cohort_word in enumerate(phoneme_words_all_temp):
                if len(cohort_word) > k and cohort_word[k] == phoneme_word[k]:
                    res.append(cohort_word)
            
            phoneme_words_all_temp = res[:]
            
            # Update probabilities for next phoneme position
            for w, cohort_word in enumerate(phoneme_words_all_temp):
                if len(cohort_word) <= k + 1:
                    continue
                
                # Get frequency of this word
                cohort_word_str = ' '.join(cohort_word)
                if cohort_word_str in df_word_grapheme['phoneme'].to_list():
                    index = df_word_grapheme.loc[df_word_grapheme['phoneme'] == cohort_word_str].index[0]
                    grapheme = df_word_grapheme['grapheme'][index]
                    freq_match = df_freq.loc[df_freq['Word'] == grapheme]
                    
                    if len(freq_match) == 0:
                        freq = 1 / num_words_all_cohort
                    else:
                        freq = float(freq_match.iloc[-1]['SUBTLEXWF'])
                else:
                    freq = 1 / num_words_all_cohort
                
                next_phone_idx = phone2int[cohort_word[k + 1]]
                Prob_matrix[k + 1, next_phone_idx] += freq / num_words_all_cohort
        
        # Calculate entropy and surprisal for each phoneme
        for k in range(len(phoneme_word)):
            shannon_k = -np.sum(Prob_matrix[k] * np.log2(Prob_matrix[k]))
            phone_idx = phone2int[phoneme_word[k]]
            surprisal_k = -np.log2(Prob_matrix[k, phone_idx] / Prob_matrix[k].sum())
            
            shannon_all.append(shannon_k)
            surprisal_all.append(surprisal_k)
            
            # Get word frequency
            grapheme_lower = grapheme_words_test[i].lower()
            ff = df_freq.loc[df_freq['Word'] == grapheme_lower]
            if len(ff) > 0:
                word_freq = -np.log2(ff.iloc[-1]['SUBTLEXWF'] / num_words_all_cohort)
            else:
                word_freq = 'NAN'
                print(f"Word '{grapheme_words_test[i]}' does not have a frequency")
            
            word_freq_in_story_all.append(word_freq)
    
    return shannon_all, surprisal_all, word_freq_in_story_all


def main():
    """Main execution function."""
    
    language = 'Dutch'
    Condition = 'Word_list' # Choose Sentences or Word_list
    
    # Paths
    root = Path.cwd().parents[0]
    dict_folder = root / 'Materials' 
       
    dict_filename = 'Dutch_dict_2022.txt'
    freq_filename = 'SUBTLEX-NL_filtered_2022_cut.csv'
    
    dict_path = os.path.join(dict_folder, dict_filename)
    freq_file_path = os.path.join(dict_folder, freq_filename)
    
    # Set up phoneme file paths
    if Condition == 'Sentences':
        phoneme_folder = dict_folder / 'Stimuli' / 'Transription' / 'Word_phoneme_transcription_of_sentences'
    else:
        phoneme_folder = dict_folder / 'Stimuli' / 'Transription' / 'Word_phoneme_transcription_of_words'
    phoneme_files = [f for f in os.listdir(phoneme_folder) if f.endswith('.csv')]
    
    # Create output folder
    output_folder = dict_folder / 'Cohort_model' / Condition
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)   
    
    
    # Load data
    print("Loading dictionary and frequency data...")
    df_freq = pd.read_csv(freq_file_path, sep=';')
    all_words = load_text_data(dict_path, language)
    
    # Create phoneme vocabulary
    print("Creating phoneme vocabulary...")
    Dictionary = PrononVocab()
    Dictionary(all_words)
    
    # Create phoneme mappings
    unique_phones = list(set(Dictionary.all_phones))
    for i, phone in enumerate(unique_phones):
        Dictionary.phone2int[phone] = i
        Dictionary.int2phone[i] = phone
    
    num_phones = len(Dictionary.phone2int)
    
    # Save phoneme mappings
    with open(os.path.join(dict_folder, 'phone2int_cohort.pkl'), "wb") as f:
        pickle.dump(Dictionary.phone2int, f)
    
    with open(os.path.join(dict_folder, 'int2phone_cohort.pkl'), "wb") as f:
        pickle.dump(Dictionary.int2phone, f)
    
    # Process frequency data
    print("Processing frequency data...")
    num_words = len(df_freq['Word'])
    num_words_all_cohort = 970843  # Total words in Dutch cohort
    
    # Initialize phoneme counter
    Counter = np.ones(num_phones) / num_words_all_cohort
    
    # Create grapheme-phoneme DataFrame
    df_word_grapheme = pd.DataFrame(all_words, columns=['grapheme', 'phoneme', 'delete'])
    df_word_grapheme = df_word_grapheme.drop(columns='delete')
    
    # Extract phoneme transcriptions for words in frequency file
    phoneme_words_all = []
    graph_words_all = []
    no_phoneme_transcription_count = 0
    
    for w in range(num_words):
        word = df_freq['Word'][w]
        if word in df_word_grapheme['grapheme'].to_list():
            graph_words_all.append(word)
            freq = float(df_freq['SUBTLEXWF'][w])
            index = df_word_grapheme.loc[df_word_grapheme['grapheme'] == word].index[0]
            phoneme_word = df_word_grapheme['phoneme'][index]
            phoneme_list = phoneme_word.split(' ')
            phoneme_words_all.append(phoneme_list)
            
            # Update counter for first phoneme
            first_phone_idx = Dictionary.phone2int[phoneme_list[0]]
            Counter[first_phone_idx] += freq / num_words_all_cohort
        else:
            no_phoneme_transcription_count += 1
            print(f"No phoneme transcription: {word} (index {w})")
    

    
    # Process each phoneme file
    print(f"Processing {len(phoneme_files)} phoneme files...")
    for p, phoneme_file in enumerate(phoneme_files):
        print(f"Processing file {p+1}/{len(phoneme_files)}: {phoneme_file}")
        
        df_phonemes = pd.read_table(
            os.path.join(phoneme_folder, phoneme_file),
            encoding='utf-8',
            sep=','
        )
        
        # Extract phoneme transcriptions for words in story
        j = 0
        phonemes_words_test = []
        grapheme_words_test = []
        
        while j < len(df_phonemes):
            phoneme_word = []
            word = df_phonemes['words'][j]
            
            while j < len(df_phonemes) and df_phonemes['words'][j] == word:
                phoneme_word.append(df_phonemes['phonemes'][j])
                j += 1
            
            phonemes_words_test.append(phoneme_word)
            grapheme_words_test.append(word)
        
        # Calculate cohort metrics
        shannon_all, surprisal_all, word_freq_all = calculate_cohort_metrics(
            phonemes_words_test, grapheme_words_test,
            phoneme_words_all, df_word_grapheme, df_freq,
            num_words_all_cohort, Dictionary.phone2int, Counter, num_phones
        )
        
        # Save results
        Data_cohort = {
            'cohort_entropy': shannon_all,
            'cohort_surprisal': surprisal_all,
            'word_freq': word_freq_all
        }
        df_cohort = pd.DataFrame(data=Data_cohort)
        df_all = pd.concat([df_phonemes, df_cohort], axis=1)
        
        output_filename = phoneme_file[:-4] + '_cohort_model.csv'
        output_path = os.path.join(output_folder, output_filename)
        df_all.to_csv(output_path, index=False, sep=';', line_terminator='\n')
    
    print("Processing complete!")


if __name__ == "__main__":
    main()