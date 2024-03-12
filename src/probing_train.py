import gc
import os
import psutil
import warnings
import logging as log
warnings.filterwarnings('ignore')
log.getLogger(__name__).setLevel(log.INFO)

import torch
import hydra
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

import utils
from model.pooling import choose_pooler
from model.probing_pair import ProbingPair
from model.probe import choose_probe
from model.subject import HuggingFace
from get_spans import choose_span_getter

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f'Using device: {device}')

def build_probing_model(cfg):
    '''
    Build the probing model using the configuration files
    '''
    log.info('Building probing model...')
    subject = HuggingFace(cfg.model.model_path).to(device)
    pooler = choose_pooler(cfg).to(device)
    probe = choose_probe(cfg).to(device)
    return ProbingPair(subject, pooler, probe).to(device)

def load_tokenizer(cfg):
    '''
    Load the tokenizer from the model path
    '''
    log.info('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_data(cfg):
    '''
    Load the data from the path
    '''
    log.info('Loading data...')
    train = pd.read_parquet(cfg.task.train_file)
    val = pd.read_parquet(cfg.task.val_file)
    test = pd.read_parquet(cfg.task.test_file)
    return train, val, test

def training_modules(model, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.task.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = utils.EarlyStopper(patience=cfg.task.early_stopping_patience, min_delta=cfg.task.min_delta)
    return optimizer, criterion, early_stopper

def train_loop(model, cfg, train, tokenizer, optimizer, criterion):
    '''
    Main training loop (for each epoch)
    '''
    data = train.sample(frac=cfg.task.train_size).reset_index(drop=True)
    model.train()
    stats = utils.StatTracker()
    span_getter = choose_span_getter(cfg)
    for batch in tqdm(np.array_split(data, len(data) // cfg.task.batch_size), desc='Training'):
        optimizer.zero_grad()
        inputs = utils.encode_batch(batch, tokenizer).to(device)
        target_spans = torch.tensor(span_getter.get_span_batch(batch, tokenizer), device=device)
        target_labels = torch.tensor(span_getter.get_labels(batch), device=device)
        outputs = model([inputs, target_spans])
        loss = criterion(outputs, target_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.task.gradient_clip)
        optimizer.step()
        outputs_softmax = torch.nn.functional.softmax(outputs, dim=1)
        acc = (outputs_softmax.argmax(1) == target_labels).sum().item()
        f1 = f1_score(target_labels.cpu().numpy(), outputs_softmax.argmax(1).cpu().numpy(), average='micro')
        stats.update(loss.item(), acc, len(target_labels), f1)
    return stats.get_stats()

def val_test_loop(model, cfg, data, tokenizer, criterion):
    '''
    Main validation and test loop
    '''
    model.eval()
    stats = utils.StatTracker()
    span_getter = choose_span_getter(cfg)

    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024 / 1024
    gpu_mem_usage = torch.cuda.memory_allocated() / 1024 / 1024
    #print(mem_usage, gpu_mem_usage)
    count = -1

    with torch.no_grad():
        for batch in tqdm(np.array_split(data, len(data) // cfg.task.batch_size), desc='Validation'):
            inputs = utils.encode_batch(batch, tokenizer).to(device)
            target_spans = torch.tensor(span_getter.get_span_batch(batch, tokenizer), device=device)
            target_labels = torch.tensor(span_getter.get_labels(batch), device=device)
            # if count % 100 == 0 or count == 363 or count == 364:
            #     print(inputs['input_ids'].shape)
            outputs = model([inputs, target_spans])
            loss = criterion(outputs, target_labels).detach()
            outputs_softmax = torch.nn.functional.softmax(outputs, dim=1)
            acc = (outputs_softmax.argmax(1) == target_labels).sum().item()
            f1 = f1_score(target_labels.cpu().numpy(), outputs_softmax.argmax(1).cpu().numpy(), average='micro')
            stats.update(loss.item(), acc, len(target_labels), f1)

            # torch.cuda.empty_cache()
            # tensor_count = 0
            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             tensor_count += 1
            #     except:
            #         pass
            # count += 1
            # if count % 100 == 0 or count == 363:
            #     print('---Step--- : ', count)
            #     print("---Tensor count : ", tensor_count)
            #     mem_usage_after = process.memory_info().rss / 1024 / 1024
            #     gpu_mem_usage_after = torch.cuda.memory_allocated() / 1024 / 1024
            #     print(f"---Memory usage : {mem_usage_after}")
            #     print(f"---GPU memory usage :{gpu_mem_usage_after}")
            #     print(f"---Max GPU memory usage : {torch.cuda.max_memory_allocated() / 1024 / 1024}")
            #     mem_usage = mem_usage_after
            #     gpu_mem_usage = gpu_mem_usage_after

    return stats.get_stats()

@hydra.main(config_path='../config/', config_name='main')
def main(cfg):
    model = build_probing_model(cfg)
    tokenizer = load_tokenizer(cfg)
    train, val, test = load_data(cfg)
    optimizer, criterion, early_stopper = training_modules(model, cfg)
    wandb_logger = utils.WandbLogger(cfg)

    log.info('Starting training...')
    for epoch in range(cfg.task.epochs):
        train_loss, train_acc, train_f1 = train_loop(model, cfg, train, tokenizer, optimizer, criterion)
        val_loss, val_acc, val_f1 = val_test_loop(model, cfg, val, tokenizer, criterion)
        log.info(f'Epoch {epoch + 1}/{cfg.task.epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Train F1: {train_f1:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}')
        wandb_logger.log({'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1, 'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1})
        if early_stopper.early_stop(val_loss):
            log.info(f'Early stopping at epoch {epoch + 1}')
            break
    
    log.info('Testing...')
    test_loss, test_acc, test_f1 = val_test_loop(model, cfg, test, tokenizer, criterion)
    wandb_logger.log({'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1})
    log.info(f'Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f} - Test F1: {test_f1:.4f}')

if __name__ == "__main__":
    main()