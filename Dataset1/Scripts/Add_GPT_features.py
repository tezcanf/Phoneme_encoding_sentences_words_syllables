# -*- coding: utf-8 -*-
"""
GPT-2 Surprisal and Entropy Calculation for MEG Word Lists

Calculates word-level surprisal and entropy values using Dutch GPT-2 models
for phoneme-transcribed word lists.

@author: filiztezcan
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from pathlib import Path

# Suppress transformers logging
transformers.logging.get_verbosity = lambda: logging.NOTSET


# ============================================================================
# Configuration
# ============================================================================
# Paths
root = Path.cwd().parents[1]

DATA_FOLDER =  root / 'Materials' / 'Cohort_model' / 'Word_list' 
OUTPUT_FOLDER = root / 'Materials' / 'GPT2' / 'Word_list' 

# Model selection
MODEL_NAME = "yhavinga/gpt2-large-dutch" 


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_csv_data(filename):
    """
    Load phoneme-transcribed word data from CSV file.
    
    Parameters
    ----------
    filename : str
        Path to CSV file with columns: 'words', 'phonemes'
    
    Returns
    -------
    grapheme_words : list of str
        List of words in graphemic form
    phonemes_words : list of list of str
        List of phoneme sequences for each word
    df_phonemes : pd.DataFrame
        Original dataframe with all phoneme data
    """
    df_phonemes = pd.read_table(filename, encoding="utf-8", sep=';')
    
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
    
    return grapheme_words, phonemes_words, df_phonemes


# ============================================================================
# Model Initialization
# ============================================================================

def load_gpt2_model(model_name):
    """
    Load GPT-2 tokenizer and model.
    
    Parameters
    ----------
    model_name : str
        HuggingFace model identifier
    
    Returns
    -------
    tokenizer : AutoTokenizer
        Loaded tokenizer
    model : AutoModelForCausalLM
        Loaded language model
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    
    return tokenizer, model


# ============================================================================
# Surprisal Calculation
# ============================================================================

def calculate_surprisal_and_entropy(graph_words, phonemes_words, tokenizer, model):
    """
    Calculate word-level surprisal and entropy using GPT-2.
    
    Parameters
    ----------
    graph_words : list of str
        Words in graphemic form
    phonemes_words : list of list of str
        Phoneme sequences for each word
    tokenizer : AutoTokenizer
        GPT-2 tokenizer
    model : AutoModelForCausalLM
        GPT-2 model
    
    Returns
    -------
    shannon_entropy : list of float
        Entropy values (on first phoneme only)
    surprisal : list of float
        Surprisal values (on first phoneme only)
    word_marker : list of int
        Binary marker (1 for first phoneme, 0 otherwise)
    """
    shannon_entropy = []
    surprisal_values = []
    word_marker = []
    
    context = ['begin']
    
    for w, word in enumerate(graph_words):
        # Prepare context and test sequences
        context_text = ' '.join(context)
        test_text = ' '.join(context + [word])
        
        # Tokenize
        input_ids_context = tokenizer.encode(context_text, return_tensors="pt")
        input_ids_test = tokenizer.encode(test_text, return_tensors="pt")
        
        # Get logits for next token prediction
        with torch.no_grad():
            next_token_logits = model(input_ids_context, return_dict=True).logits[:, -1, :]
        
        # Get first token of target word
        len_new_tokens = input_ids_test.shape[1] - input_ids_context.shape[1]
        first_token_id = input_ids_test[0, -len_new_tokens].item()
        
        # Calculate probabilities
        probs = F.softmax(next_token_logits, dim=-1)
        probs_np = probs.detach().numpy()[0]
        
        # Calculate entropy (Shannon entropy)
        entropy = -np.sum(probs_np * np.log2(probs_np + 1e-10))
        
        # Calculate surprisal for first token
        word_surprisal = -np.log2(probs_np[first_token_id] + 1e-10)
        
        # Store values: only on first phoneme of each word
        for ph_idx in range(len(phonemes_words[w])):
            if ph_idx == 0:
                shannon_entropy.append(entropy)
                surprisal_values.append(word_surprisal)
                word_marker.append(1)
            else:
                shannon_entropy.append(0)
                surprisal_values.append(0)
                word_marker.append(0)
        
        # Update context (keep only last word as context)
        context = [word]
    
    return shannon_entropy, surprisal_values, word_marker


# ============================================================================
# Main Processing
# ============================================================================

def process_all_files():
    """
    Process all CSV files in the data folder and save results.
    """
    # Load model once
    tokenizer, model = load_gpt2_model(MODEL_NAME)
    
    # Get all CSV files
    phoneme_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    
    print(f"Found {len(phoneme_files)} files to process")
    
    for file_idx, phoneme_file in enumerate(phoneme_files):
        print(f"\nProcessing file {file_idx + 1}/{len(phoneme_files)}: {phoneme_file}")
        
        # Load data
        input_path = os.path.join(DATA_FOLDER, phoneme_file)
        graph_words, phonemes_words, df_phonemes = load_csv_data(input_path)
        
        print(f"  Words to process: {len(graph_words)}")
        
        # Calculate surprisal and entropy
        shannon_entropy, surprisal, word_marker = calculate_surprisal_and_entropy(
            graph_words, phonemes_words, tokenizer, model
        )
        
        # Create output dataframe
        df_word = pd.DataFrame({
            'word_entropy_GPT': shannon_entropy,
            'word_surprisal_GPT': surprisal,
            'word_number': word_marker,
        })
        
        # Combine with original phoneme data
        df_all = pd.concat([df_phonemes, df_word], axis=1)
        
        # Save results
        output_filename = phoneme_file.replace('.csv', '_GPT_new_large.csv')
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        df_all.to_csv(output_path, index=False, sep=';', line_terminator='\n')
        print(f"  Saved: {output_filename}")
    
    print("\nAll files processed successfully!")


# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Process all files
    process_all_files()