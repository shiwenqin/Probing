import wandb
import pandas as pd
import numpy as np
import logging as log
import torch
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model.pooling import AttentionPooler
from model.probing_pair import ProbingPair
from model.mlp import MLP
from model.bert import BERTHuggingFace
from train_utils import EarlyStopper, encode_batch, get_word_spans

from transformers import BertTokenizer

log.basicConfig(level=log.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f'Using device: {device}')

# Configuration 
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--base_model', type=str, default='bert-base-uncased', help='Base model to use')
parser.add_argument('-l', '--layer_to_probe', type=int, default=12, help='Number of layers in the base model')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('-t', '--train_percentage', type=float, default=0.5, help='Percentage of training data to use each epoch')
args = parser.parse_args()

# Model Configuration
BERT_MODEL = args.base_model
NUM_LAYERS = 12
POOLING_INPUT_DIM = 768
POOLING_HIDDEN_DIM = 256
MLP_HIDDEN_DIM = 256
LAYER_TO_PROBE = args.layer_to_probe
SINGLE_SPAN = True
NUM_CLASSES = 51

# Data Configuration
TRAIN_FILE = '../dataset/ontonotesv5_english_v12_train_processed.parquet'
VAL_FILE = '../dataset/ontonotesv5_english_v12_validation_processed.parquet'
TEST_FILE = '../dataset/ontonotesv5_english_v12_test_processed.parquet'
TRAIN_PERCENTAGE = args.train_percentage

# Training Configuration
BATCH_SIZE = args.batch_size
LEARNING_RATE = 1e-4
EPOCHS = args.epochs
GRADIENT_CLIP = 5.0
EARLY_STOP_PATIENCE = 5

wandb.init(project='pos-probing-exp', 
           config={'base_model': BERT_MODEL,
                   'num_layers': NUM_LAYERS,
                   'pooling_input_dim': POOLING_INPUT_DIM,
                   'pooling_hidden_dim': POOLING_HIDDEN_DIM,
                   'mlp_hidden_dim': MLP_HIDDEN_DIM,
                   'layer_to_probe': LAYER_TO_PROBE,
                   'single_span': SINGLE_SPAN,
                   'num_classes': NUM_CLASSES,
                   'batch_size': BATCH_SIZE,
                   'learning_rate': LEARNING_RATE,
                   'epochs': EPOCHS,
                   'gradient_clip': GRADIENT_CLIP,
                   'early_stop_patience': EARLY_STOP_PATIENCE,})

# Load models
log.info('Loading models...')
subject_model = BERTHuggingFace(BERT_MODEL, NUM_LAYERS).to(device)
pooling_model = AttentionPooler(POOLING_INPUT_DIM, POOLING_HIDDEN_DIM, LAYER_TO_PROBE, SINGLE_SPAN).to(device)
mlp_model = MLP(POOLING_HIDDEN_DIM, MLP_HIDDEN_DIM, NUM_CLASSES, dropout=0.3, single_span=SINGLE_SPAN).to(device)
probing_model = ProbingPair(subject_model, pooling_model, mlp_model).to(device)

# Load tokenizer
log.info('Loading tokenizer...')
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

# Load data
log.info('Loading data...')
train_df = pd.read_parquet(TRAIN_FILE, columns=['sentence','pos_tags'])
val_df = pd.read_parquet(VAL_FILE, columns=['sentence','pos_tags'])
test_df = pd.read_parquet(TEST_FILE, columns=['sentence','pos_tags'])

# Training Setup
log.info('Setting up training...')
optimizer = torch.optim.AdamW(probing_model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# Define Training Loop
log.info('Starting training...')

def epoch_step(model, criterion, optimizer, train_data, val_data, tokenizer, device):
    train_data = train_data.sample(frac=TRAIN_PERCENTAGE).reset_index(drop=True)
    model.train()
    train_loss = 0
    train_acc = 0
    train_num = 0
    for batch in tqdm(np.array_split(train_data, len(train_data) // BATCH_SIZE), desc='Training'):
        optimizer.zero_grad()
        inputs = encode_batch(batch, tokenizer).to(device)
        target_spans = get_word_spans(batch['sentence'], tokenizer)
        target_spans = torch.tensor(target_spans, device=device)
        target_labels = torch.tensor(np.concatenate(batch['pos_tags'].values), device=device)
        train_num += len(target_labels)
        outputs = model([inputs, target_spans])
        loss = criterion(outputs, target_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        train_loss += loss.item()
        outputs_softmax = torch.nn.functional.softmax(outputs, dim=1)
        train_acc += (outputs_softmax.argmax(1) == target_labels).sum().item()

    train_loss /= len(train_data)
    train_acc /= train_num
    model.eval()
    val_loss = 0
    val_acc = 0
    val_num = 0 
    for batch in tqdm(np.array_split(val_data, len(val_data) // BATCH_SIZE), desc='Validation'):
        inputs = encode_batch(batch, tokenizer).to(device)
        target_spans = get_word_spans(batch['sentence'], tokenizer)
        target_spans = torch.tensor(target_spans, device=device)
        target_labels = torch.tensor(np.concatenate(batch['pos_tags'].values), device=device)
        val_num += len(target_labels)
        outputs = model([inputs, target_spans])
        loss = criterion(outputs, target_labels)
        val_loss += loss.item()
        outputs_softmax = torch.nn.functional.softmax(outputs, dim=1)
        val_acc += (outputs_softmax.argmax(1) == target_labels).sum().item()
    val_loss /= len(val_data)
    val_acc /= val_num
    return train_loss, train_acc, val_loss, val_acc

# Training Loop
early_stopper = EarlyStopper(patience=EARLY_STOP_PATIENCE, min_delta=0.001)
for epoch in range(EPOCHS):
    train_loss, train_acc, val_loss, val_acc = epoch_step(probing_model, criterion, optimizer, train_df, val_df, tokenizer, device)
    log.info(f'Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
    wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
    if early_stopper.early_stop(val_loss):
        log.info(f'Early stopping at epoch {epoch + 1}')
        break

# Testing
log.info('Testing...')
probing_model.eval()
test_loss = 0
test_acc = 0
test_num = 0
for batch in tqdm(np.array_split(test_df, len(test_df) // BATCH_SIZE), desc='Testing'):
    inputs = encode_batch(batch, tokenizer).to(device)
    target_spans = get_word_spans(batch['sentence'], tokenizer)
    target_spans = torch.tensor(target_spans, device=device)
    target_labels = torch.tensor(np.concatenate(batch['pos_tags'].values), device=device)
    test_num += len(target_labels)
    outputs = probing_model([inputs, target_spans])
    loss = criterion(outputs, target_labels)
    test_loss += loss.item()
    outputs_softmax = torch.nn.functional.softmax(outputs, dim=1)
    test_acc += (outputs_softmax.argmax(1) == target_labels).sum().item()

test_loss /= len(test_df)
test_acc /= test_num
log.info(f'Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}')
wandb.log({'test_loss': test_loss, 'test_acc': test_acc})