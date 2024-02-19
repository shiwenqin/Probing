from transformers import BertTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 128

TRAIN_FILE = '../dataset/ontonotesv5_english_v12_train_processed.parquet'
VAL_FILE = '../dataset/ontonotesv5_english_v12_validation_processed.parquet'
TEST_FILE = '../dataset/ontonotesv5_english_v12_test_processed.parquet'

train_df = pd.read_parquet(TRAIN_FILE, columns=['sentence', 'pos_index', 'pos_label'])
val_df = pd.read_parquet(VAL_FILE, columns=['sentence', 'pos_index', 'pos_label'])
test_df = pd.read_parquet(TEST_FILE, columns=['sentence', 'pos_index', 'pos_label'])

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_sentence(sentence, tokenizer):
    tokenized_sentence = tokenizer.tokenize(sentence)
    full_word_indexes = []
    word_index = -1
    for token in tokenized_sentence:
        if token.startswith("##"):
            full_word_indexes.append(word_index)
        else:
            word_index += 1
            full_word_indexes.append(word_index)
    return full_word_indexes

def find_first_last_occurrences(lst, element, i):
    first_occurrence = lst.index(element) if element in lst else -1
    last_occurrence = len(lst) - 1 - lst[::-1].index(element) if element in lst else -1
    return i, first_occurrence, last_occurrence + 1

def get_span_batch(sentences, tokenizer, indexes):
    spans = []
    for i, sentence in enumerate(sentences):
        full_word_index = tokenize_sentence(sentence, tokenizer)
        span = find_first_last_occurrences(full_word_index, indexes[i], i)
        spans.append(span)
    return spans

def process(df, tokenizer, path):
    total_spans = []
    for batch in tqdm(np.array_split(df, len(df) // BATCH_SIZE)):

        target_spans = get_span_batch(batch['sentence'], tokenizer, batch['pos_index'].tolist())
        target_spans = torch.tensor(target_spans, device=device)
        print(target_spans)
        raise
        total_spans += target_spans[0].tolist()

    train_df[f'{MODEL_NAME}_target_spans'] = total_spans

    df.to_csv(path.replace('.parquet', '.csv'), index=False)
    df.to_parquet(path, index=False)

process(train_df, tokenizer, TRAIN_FILE)