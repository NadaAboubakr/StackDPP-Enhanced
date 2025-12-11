# StackDPP

## Overview
StackDPP is a stacking ensemble model for DNA-binding protein prediction that achieved 92.70% accuracy on independent tests using 452 handcrafted features and traditional machine learning methods. This repository extends the original StackDPP framework by exploring whether modern deep learning techniques can outperform the baseline. We implement three enhancement strategies: (1) a custom Transformer architecture, (2) a BiLSTM with attention mechanism, and (3) ESM-2 protein language model embeddings integrated with the original StackDPP ensemble architecture.

 
The resources attached to this repository are as follows:
  1. DataSet: This folder contains the datasets used in this work. pdb1075.fasta, pdb1035.fasta, pdb186.fasta are datasets from previous work and uniprot1424.fasta, uniprot356.fasta are the proposed new benchmark datasets.
  2. Features: This folder contains the finally selected features for StackDPP.
  3. Results: The results of some of our experiments are placed under this folder as CSV files. All the experimental results for the original model are available in the manuscript.
  4. Scripts: Run the script in the script folder to generate results. Both the notebook version and Python script execute the same logic.
  5. Models/Uniprot1424: These are some trained models on Uniprot1424 dataset. The models tied to previous literature are according to our implementation of their methodologies.
     
 ## Run the predictor on your protein sequences
  1. We will need the sequence, PSSM, and Spider output of the sequence. Two example files, example.pssm and example.spd33 have been uploaded.
  2. Run the TestANewSequence script (either Python script or notebook version) by setting up the variables (sequence, pssmFile, spiderFile).

  

## Enhanced Models and Additional Scripts

This repository has been extended with three additional deep learning approaches for DNA-binding protein prediction:

### 1. BiLSTM Model (BiLSTM.py)
A Bidirectional LSTM model with attention mechanism for sequence-based prediction.

**How to run:**
```bash
python Scripts/BiLSTM.py
```

**Requirements:** The model automatically loads features from `Features/rf452.npz` and trains on the training/test split.

### 2. Transformer Model (Transformer-PseAAC.py)
A transformer-based architecture leveraging self-attention for DNA-binding protein classification.

**How to run:**
```bash
python Scripts/Transformer-PseAAC.py
```

### 3. ESM-2 Enhanced Model (ESM2-Stackdpp enhancement.ipynb)
Integration of ESM-2 protein language model embeddings with the original StackDPP features.

```bash
jupyter notebook "Scripts/ESM2-Stackdpp enhancement.ipynb"
```

**Input requirements:**
- Training sequences: `Dataset/uniprot1424.fasta`
- Test sequences: `Dataset/uniprot356.fasta`
- Original features: `Features/rf452.npz`



**Note:** The ESM-2 model requires downloading the pre-trained weights on first run (approximately 135MB for the `esm2_t12_35M_UR50D` model).

## Installation

Install all required dependencies:
```bash
pip install -r requirements.txt
```



