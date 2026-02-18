"""
Data loading utilities for TRF estimation pipeline.
Handles stimulus ordering, block order, and rejected epoch detection.
"""
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import mne

from config import (
    PATH_BLOCK,
    CONDITION_TO_EPOCH_NAME,
    BLOCK_ORDER_CONDITION_NAMES,
    EVENT_DICT_CONDITIONS,
)


def get_stimuli_list(stimulus_dir: Path) -> List[str]:
    """
    Get sorted list of stimulus names from directory.
    
    Args:
        stimulus_dir: Directory containing stimulus .wav files
        
    Returns:
        Sorted list of stimulus names (without .wav extension)
    """
    stimuli = [
        f.split('.')[0] 
        for f in os.listdir(stimulus_dir) 
        if f.endswith('wav')
    ]
    stimuli.sort()
    return stimuli


def get_block_order() -> List[str]:
    """
    Get sorted list of block order files.
    
    Returns:
        Sorted list of block order file names
    """
    block_order = [
        f.split('.')[0] 
        for f in os.listdir(PATH_BLOCK) 
        if f.startswith('Block')
    ]
    block_order.sort()
    return block_order


def get_subject_block_number(subject: str) -> int:
    """
    Get the block number assigned to a subject.
    
    Args:
        subject: Subject ID
        
    Returns:
        Block number for the subject
    """
    df_no = pd.read_excel(
        PATH_BLOCK / 'subjects_block_no.xlsx',
        header=None
    )
    block_no = df_no[df_no[0] == subject][1].values[0]
    return int(block_no)


def get_stimulus_instances(
    condition: str,
    block_no: int,
    block_order: List[str]
) -> List[int]:
    """
    Get stimulus instance numbers for a condition based on block order.
    
    Args:
        condition: Condition name ('sentences', 'words', or 'syllables')
        block_no: Block number for the subject
        block_order: List of block order file names
        
    Returns:
        List of stimulus instance numbers
    """
    # Map to original directory name first
    from config import CONDITION_DIR_MAPPING
    dir_name = CONDITION_DIR_MAPPING.get(condition, condition)
    
    # Then handle special case for block order files
    search_condition = BLOCK_ORDER_CONDITION_NAMES.get(condition, dir_name)
    
    instance_no = []
    for t in range(10):
        i = (t + block_no) % 10
        if i == 0:
            i = 10
            
        df = pd.read_excel(PATH_BLOCK / f'{block_order[i-1]}.xlsx')
        for j in range(len(df)):
            if df['Block_no'][j] == search_condition:
                instance_no.append(int(df['instance'][j]))
                
    return instance_no


def get_subject_stimuli(
    condition: str,
    subject: str,
    stimulus_dir: Path
) -> List[str]:
    """
    Get the list of stimuli for a subject in the correct presentation order.
    
    Args:
        condition: Condition name
        subject: Subject ID
        stimulus_dir: Directory containing stimulus files
        
    Returns:
        List of stimulus names in presentation order
    """
    stimuli = get_stimuli_list(stimulus_dir)
    block_order = get_block_order()
    block_no = get_subject_block_number(subject)
    instance_no = get_stimulus_instances(condition, block_no, block_order)
    
    stimuli_subject = [stimuli[ins] for ins in instance_no]
    return stimuli_subject


def find_rejected_epochs(
    epochs: mne.Epochs,
    events_orig: np.ndarray,
    condition: str,
    stimuli_subject: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Identify rejected epochs and remove corresponding stimuli.
    
    Args:
        epochs: MNE Epochs object
        events_orig: Original events array
        condition: Condition name
        stimuli_subject: List of stimulus names in presentation order
        
    Returns:
        Tuple of (filtered events array, filtered stimuli list)
    """
    # Get epoch name for this condition
    epoch_name = CONDITION_TO_EPOCH_NAME.get(condition)
    if epoch_name is None:
        epoch_name = ''.join(condition.split('_'))
    
    # Get event ID for this condition
    event_id = EVENT_DICT_CONDITIONS[epoch_name]
    
    # Filter original events for this condition
    events_org_cond = []
    for tr in range(len(events_orig)):
        if events_orig[tr, 2] == event_id:
            events_org_cond.append(events_orig[tr])
    
    events_org_cond = np.array(events_org_cond)
    
    # Get events from epochs (these exclude rejected epochs)
    events_cond = epochs[epoch_name].events
    
    # Create mutable copy of stimuli list
    stimuli_filtered = stimuli_subject.copy()
    
    # Find and remove rejected epochs
    p = 0
    rejected_epoch = 0
    rejected_indices = []
    
    while p < len(events_org_cond):
        # Special case: if we're at the last epoch and it's rejected
        if (len(events_org_cond) == 20 and 
            len(events_cond) == 19 and 
            rejected_epoch == 19):
            events_org_cond = np.delete(events_org_cond, p, 0)
            rejected_indices.append(rejected_epoch)
            print(f"Rejected epoch: {rejected_epoch}")
            break
            
        # Check if current epoch was kept or rejected
        if events_org_cond[p, 0] == events_cond[p - len(rejected_indices), 0]:
            # Epoch was kept
            rejected_epoch += 1
            p += 1
        else:
            # Epoch was rejected
            events_org_cond = np.delete(events_org_cond, p, 0)
            rejected_indices.append(rejected_epoch)
            print(f"Rejected epoch: {rejected_epoch}")
            rejected_epoch += 1
    
    # Remove rejected stimuli
    for idx in sorted(rejected_indices, reverse=True):
        stimuli_filtered.pop(idx)
    
    return events_org_cond, stimuli_filtered


def load_and_filter_epochs(
    epochs_path: Path,
    events_path: Path,
    condition: str,
    stimuli_subject: List[str]
) -> Tuple[mne.Epochs, List[str], mne.Info]:
    """
    Load epochs, identify rejected trials, and filter stimuli list.
    
    Args:
        epochs_path: Path to epochs file
        events_path: Path to events file
        condition: Condition name
        stimuli_subject: Initial list of stimuli
        
    Returns:
        Tuple of (epochs object, filtered stimuli list, info object)
    """
    epochs = mne.read_epochs(str(epochs_path), preload=True)
    events_orig = mne.read_events(str(events_path))
    
    events_filtered, stimuli_filtered = find_rejected_epochs(
        epochs, events_orig, condition, stimuli_subject
    )
    
    return epochs, stimuli_filtered, epochs.info
