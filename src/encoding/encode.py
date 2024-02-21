import pickle
import argparse
import warnings
import gc
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
parser.add_argument('-m', '--base_model', type=str, default='bert-base-uncased', help='Base model to use')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for encoding data')
args = parser.parse_args()

# Model Configuration
BASE_MODEL = args.base_model
LAYER_NUM = 13 # 12 hidden layers + 1 embedding layer

# Data Configuration
TRAIN_FILE = '../../dataset/ontonotesv5_english_v12_train_processed.parquet'
# VAL_FILE = '../../dataset/ontonotesv5_english_v12_validation_processed.parquet'
# TEST_FILE = '../../dataset/ontonotesv5_english_v12_test_processed.parquet'

BATCH_SIZE = args.batch_size

# Load Models
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModel.from_pretrained(BASE_MODEL).to(device)

# Load Data
train_df = pd.read_parquet(TRAIN_FILE, columns=['sentence'])
# val_df = pd.read_parquet(VAL_FILE, columns=['sentence'])
# test_df = pd.read_parquet(TEST_FILE, columns=['sentence'])

max_len_data = max(train_df['sentence'].apply(lambda x: len(x.split())))
# print(max_len_data)

# Encode Data
bert_encoding = {}
for i in range(LAYER_NUM):
    bert_encoding[i] = []

model.eval()
for batch in tqdm(np.array_split(train_df, len(train_df) // BATCH_SIZE), desc='Training'):
    batch = batch.reset_index(drop=True)
    inputs = tokenizer(batch['sentence'].tolist(), return_tensors='pt', 
                       padding='max_length', add_special_tokens=False, 
                       max_length = max_len_data).to(device)
    outputs = model.forward(**inputs, output_hidden_states=True)
    for i in range(LAYER_NUM):
        bert_encoding[i].append(outputs.hidden_states[i].detach().cpu().numpy())
    # for obj in gc.get_objects():
    #     count = 0
    #     if torch.is_tensor(obj):
    #         count += 1
    # print(count)

# print(bert_encoding[0].shape)
# print(bert_encoding[1].shape)
# print(train_df.shape)

with open('bert_encoding_train.pkl', 'wb') as f:
    pickle.dump(bert_encoding, f)