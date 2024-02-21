import pickle
import argparse
import warnings
import logging as log
from tqdm import tqdm

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Basic Setup
log.basicConfig(level=log.INFO)
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Input Configuration
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--base_model', type=str, default='bert-base-uncased', help='Base model to use')
args = parser.parse_args()

# Model Configuration
BASE_MODEL = args.base_model

# Data Configuration
TRAIN_FILE = '../../dataset/ontonotesv5_english_v12_train_processed.parquet'
VAL_FILE = '../../dataset/ontonotesv5_english_v12_validation_processed.parquet'
TEST_FILE = '../../dataset/ontonotesv5_english_v12_test_processed.parquet'

# Load Models
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModel.from_pretrained(BASE_MODEL).to(device)

# Load Data
train_df = pd.read_parquet(TRAIN_FILE, columns=['sentence', 'pos_index', 'pos_label'])
val_df = pd.read_parquet(VAL_FILE, columns=['sentence', 'pos_index', 'pos_label'])
test_df = pd.read_parquet(TEST_FILE, columns=['sentence', 'pos_index', 'pos_label'])

