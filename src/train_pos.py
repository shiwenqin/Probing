import wandb
import pandas as pd
import numpy as np
import logging as log
import torch
from tqdm import tqdm

from model.pooling import AttentionPooler
from model.probing_pair import ProbingPair
from model.mlp import MLP
from model.bert import BERTHuggingFace

from transformers import BertTokenizer

log.basicConfig(level=log.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f'Using device: {device}')

# Model Configuration
BERT_MODEL = 'bert-base-uncased'
NUM_LAYERS = 12
POOLING_INPUT_DIM = 768
POOLING_HIDDEN_DIM = 256
MLP_HIDDEN_DIM = 256
LAYER_TO_PROBE = 12
SINGLE_SPAN = True
NUM_CLASSES = 51

# Data Configuration
TRAIN_FILE = '../dataset/ontonotesv5_english_v12_train_processed.parquet'
VAL_FILE = '../dataset/ontonotesv5_english_v12_validation_processed.parquet'
TEST_FILE = '../dataset/ontonotesv5_english_v12_test_processed.parquet'

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50

# Load models
log.info('Loading models...')
subject_model = BERTHuggingFace(BERT_MODEL, NUM_LAYERS).to(device)
pooling_model = AttentionPooler(POOLING_INPUT_DIM, POOLING_HIDDEN_DIM, LAYER_TO_PROBE, SINGLE_SPAN).to(device)
mlp_model = MLP(POOLING_HIDDEN_DIM, MLP_HIDDEN_DIM, NUM_CLASSES, dropout=0.3, single_span=SINGLE_SPAN).to(device)
probing_model = ProbingPair(subject_model, pooling_model, mlp_model).to(device)
probing_model = torch.compile(probing_model)

# Load tokenizer
log.info('Loading tokenizer...')
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

# Load data
log.info('Loading data...')
train_df = pd.read_parquet(TRAIN_FILE, columns=['sentence', 'pos_index', 'pos_label'])
val_df = pd.read_parquet(VAL_FILE, columns=['sentence', 'pos_index', 'pos_label'])
test_df = pd.read_parquet(TEST_FILE, columns=['sentence', 'pos_index', 'pos_label'])

# Training Setup
log.info('Setting up training...')
optimizer = torch.optim.Adam(probing_model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# Define utility functions
def encode_batch(batch, tokenizer):
    """
    :param batch: a pandas DataFrame with columns 'sentence' and 'target_spans'.
    :param tokenizer: a tokenizer object from the HuggingFace's transformers library.
    """
    inputs = tokenizer(batch['sentence'].tolist(), return_tensors='pt', padding='longest', add_special_tokens=False)
    return inputs

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

# Training Loop
log.info('Starting training...')

def epoch_step(model, criterion, optimizer, train_data, val_data, tokenizer, device):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch in tqdm(np.array_split(train_data, len(train_data) // BATCH_SIZE), desc='Training'):
        optimizer.zero_grad()
        inputs = encode_batch(batch, tokenizer).to(device)
        target_spans = get_span_batch(batch['sentence'], tokenizer, batch['pos_index'].tolist())
        target_spans = torch.tensor(target_spans, device=device)
        #print(batch['pos_label'])
        target_labels = torch.tensor(batch['pos_label'].values, device=device)
        outputs = model([inputs, target_spans])
        loss = criterion(outputs, target_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (outputs.argmax(1) == target_labels).sum().item()
        # print(outputs.argmax(1))
        # print(target_labels)
        # print(train_loss)
        # raise

    train_loss /= len(train_data)
    train_acc /= len(train_data)
    model.eval()
    val_loss = 0
    val_acc = 0
    for batch in tqdm(np.array_split(val_data, len(val_data) // BATCH_SIZE), desc='Validation'):
        inputs = encode_batch(batch, tokenizer).to(device)
        target_spans = get_span_batch(batch['sentence'], tokenizer, batch['pos_index'].tolist())
        target_spans = torch.tensor(target_spans, device=device)
        target_labels = torch.tensor(batch['pos_label'].values, device=device)
        outputs = model([inputs, target_spans])
        loss = criterion(outputs, target_labels)
        val_loss += loss.item()
        val_acc += (outputs.argmax(1) == target_labels).sum().item()
    val_loss /= len(val_data)
    val_acc /= len(val_data)
    return train_loss, train_acc, val_loss, val_acc

for epoch in range(EPOCHS):
    train_loss, train_acc, val_loss, val_acc = epoch_step(probing_model, criterion, optimizer, train_df, val_df, tokenizer, device)
    log.info(f'Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
    