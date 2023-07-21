# Prediction of CRISPR-Cas9 Off-Target Activities with Mismatches and Indels Based on Hybrid Neural Network
The repository includes a coding scheme and hybrid network model called CRISPR-HW for predicting off-target activities of insertions, deletions and mismatches in CRISPR/Cas9 gene editing.

## Prerequisite
Following Python packages should be installed:

numpy  
pandas  
scikit-learn  
TensorFlow  
Keras  
matplotlib  

## Usage
You can process the dataset by calling Coding_pairs.py to obtain the desired coding format for CRISPR-HW.  
After obtaining the coding format for the new dataset, you can train and test the dataset using CRISPR-HW.py. All the model architectures in the experiment are described in detail in model/model.py, which also contains other models for non-targeted prediction.  
The generalizability.py file is the code for testing the generalizability of the model.  
The P&N sample analysis.py file and position_type_reads.py are the code that analyzes the dataset.  

## Data Description
All dataset sources are explained in the paper.  
All datasets are in the ./datasets file, The text within /datasets/word embedding are data that we were asked to process to match our encoding.  
