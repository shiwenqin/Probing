import warnings
import os
import json
import logging as log
warnings.filterwarnings('ignore')
log.getLogger(__name__).setLevel(log.INFO)

import torch
import hydra
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
from safetensors.torch import save_file, load_file

import utils
from model.pooling import choose_pooler
from model.probing_pair import ProbingPair
from model.probe import choose_probe
from model.subject import Untrained
from architectures.crammed_bert import construct_crammed_bert
from get_spans import choose_span_getter

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f'Using device: {device}')

def build_probing_model(cfg):
    '''
    Build the probing model using the configuration files
    '''
    log.info('Building probing model...')
    with open(os.path.join(cfg.model.model_path, "model_config.json"), "r") as file:
        cfg_arch = OmegaConf.create(json.load(file))
    subject = construct_crammed_bert(cfg_arch, cfg.model.vocab_size, cfg.task.num_classes).to(device)
    subject = load_weights(cfg, subject)
    pooler = choose_pooler(cfg).to(device)
    probe = choose_probe(cfg).to(device)
    return ProbingPair(subject, pooler, probe, cfg.task.freeze_subject).to(device)

def load_weights(cfg, model):
    state_dict = load_file(os.path.join(cfg.model.model_path, "model.safetensors"))
    sanitized_state = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            sanitized_state[k[8:]] = v
        # else:
        #     sanitized_state[k] = v
    model.load_state_dict(sanitized_state)
    return model

def load_data(cfg):
    '''
    Load the data from the path
    '''
    log.info('Loading data...')
    train = pd.read_parquet(cfg.task.train_file)
    train = train[:-1]
    val = pd.read_parquet(cfg.task.val_file)
    val = val[:-1]
    test = pd.read_parquet(cfg.task.test_file)
    test = test[:-1]
    return train, val, test

def training_modules(model, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.task.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = utils.EarlyStopper(patience=cfg.task.early_stopping_patience, min_delta=cfg.task.min_delta)
    return optimizer, criterion, early_stopper

def train_loop(model, cfg, train, optimizer, criterion):
    '''
    Main training loop (for each epoch)
    '''
    data = train.sample(frac=cfg.task.train_size).reset_index(drop=True)
    #data = train
    model.train()
    stats = utils.StatTracker()
    for batch in tqdm(np.array_split(data, len(data) // cfg.task.batch_size), desc='Training'):
        optimizer.zero_grad()
        inputs = torch.tensor(np.vstack(batch['input_ids'])).to(device)
        target_spans = get_span(batch).to(device)
        target_labels = torch.tensor(np.concatenate(batch['labels'].values)).to(device)
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

def val_test_loop(model, cfg, data, criterion):
    '''
    Main validation and test loop
    '''
    model.eval()
    stats = utils.StatTracker()
    with torch.no_grad():
        for batch in tqdm(np.array_split(data, len(data) // cfg.task.batch_size), desc='Validation'):
            inputs = torch.tensor(np.vstack(batch['input_ids'])).to(device)
            target_spans = get_span(batch).to(device)
            target_labels = torch.tensor(np.concatenate(batch['labels'].values)).to(device)
            outputs = model([inputs, target_spans])
            loss = criterion(outputs, target_labels)
            outputs_softmax = torch.nn.functional.softmax(outputs, dim=1)
            acc = (outputs_softmax.argmax(1) == target_labels).sum().item()
            f1 = f1_score(target_labels.cpu().numpy(), outputs_softmax.argmax(1).cpu().numpy(), average='micro')
            stats.update(loss.item(), acc, len(target_labels), f1)
    return stats.get_stats()


def get_span(batch):
    pos_spans = []
    for index, spans in enumerate(batch['spans']):
        for span in spans:
            pos_spans.append((index, span[0], span[1]))
    return torch.tensor(pos_spans)

def save_model(model, save_path):
    '''
    Save the model to the path
    '''
    log.info(f'Saving model to {save_path}')
    save_file(model.state_dict(), save_path)

@hydra.main(config_path='../config/', config_name='main')
def main(cfg):
    model = build_probing_model(cfg)
    train, val, test = load_data(cfg)
    optimizer, criterion, _ = training_modules(model, cfg)
    wandb_logger = utils.WandbLogger(cfg)

    log.info('Starting training...')
    for epoch in range(cfg.task.epochs):
        if cfg.task.save_intermediates and cfg.save_model:
            save_path = cfg.model.save_path + f'_{epoch}.safetensor'
            save_model(model, save_path)
        train_loss, train_acc, train_f1 = train_loop(model, cfg, train, optimizer, criterion)
        val_loss, val_acc, val_f1 = val_test_loop(model, cfg, val, criterion)
        log.info(f'Epoch {epoch + 1}/{cfg.task.epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Train F1: {train_f1:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}')
        wandb_logger.log({'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1, 'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1})

    log.info('Testing...')
    test_loss, test_acc, test_f1 = val_test_loop(model, cfg, test, criterion)
    wandb_logger.log({'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1})
    log.info(f'Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f} - Test F1: {test_f1:.4f}')

    log.info('Training finished!')
    if cfg.save_model:
        save_model(model, cfg.model.save_path + '.safetensor')

if __name__ == "__main__":
    main()