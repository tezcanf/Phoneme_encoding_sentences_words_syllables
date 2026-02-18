================================================================================
QUICK START GUIDE
MEG TRF Analysis Pipeline
================================================================================

This guide will walk you through running the complete analysis pipeline from
raw MEG data to publication figures.

================================================================================
PREREQUISITES
================================================================================

✓ Python 3.8+ with conda/pip
✓ Raw MEG data in CTF format (.ds directories)
✓ Phoneme transcriptions for your stimuli
✓ Language dictionary and frequency files
✓ Audio files (.wav) of stimuli
✓ At least 32GB RAM (64GB recommended)

================================================================================
STEP 0: SETUP ENVIRONMENT
================================================================================

To create the environment:

1. With conda:
   conda env create -f environment.yml

2. Or with pip:
   pip install -r requirements.txt


Detailed steps for each Dataset can be found under Dataset folders.

================================================================================
CITATION 
================================================================================

If you use this pipeline, please cite:

Tezcan, F., Ten Oever, S., Bai, F., te Rietmolen, N., & Martin, A. (2025). 
Linguistic structure and language familiarity sharpen phoneme encoding in the brain.


================================================================================
CONTACT & SUPPORT
================================================================================

Author: Filiz Tezcan
Email: filiz.tezcansemerci@mpi.nl
Lab: MPI for Psycholinguistics