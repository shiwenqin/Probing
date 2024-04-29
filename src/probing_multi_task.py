import warnings
import os
import json
import logging as log

import torch
import hydra
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.metrics import f1_score
from safetensors.torch import save_file, load_file

import utils
from model.pooling import choose_pooler
from model.probing_pair import ProbingPair
from model.probe import choose_probe
from architectures.crammed_bert import construct_crammed_bert

warnings.filterwarnings("ignore")
log.getLogger(__name__).setLevel(log.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f'Using device: {device}')

def build_base_mdoel(cfg):
    '''
    Build the base model using the configuration files
    '''
    log.info('Building base model...')
    with open(os.path.join(cfg.model.model_path, "model_config.json"), "r") as file:
        cfg_arch = OmegaConf.create(json.load(file))
    model = construct_crammed_bert(cfg_arch, cfg.model.vocab_size).to(device)
    model = load_weights(cfg, model)
    return model

def build_probing_model(base_model, task_cfg, cfg):
    '''
    Build the probing model using the base model and task specific configuration
    '''
    pooler = choose_pooler(cfg, task_cfg).to(device)
    probe = choose_probe(cfg, task_cfg).to(device)
    return ProbingPair(base_model, pooler, probe, cfg.task.freeze_subject).to(device)

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

def load_data(task_cfg):
    '''
    Load the data from the path
    '''
    log.info('Loading data...')
    train = pd.read_parquet(task_cfg.train_file)
    train = train[:-1]
    val = pd.read_parquet(task_cfg.val_file)
    val = val[:-1]
    test = pd.read_parquet(task_cfg.test_file)
    test = test[:-1]
    return train, val, test

def training_modules(model, task_cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=task_cfg.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = utils.EarlyStopper(patience=task_cfg.early_stopping_patience, min_delta=task_cfg.min_delta)
    return optimizer, criterion, early_stopper

def get_span(batch):
    batch_spans = []
    for index, spans in enumerate(batch['spans']):
        for span in spans:
            batch_spans.append((index, span[0], span[1]))
    return torch.tensor(batch_spans)

def get_span_double(batch):
    batch_spans1 = []
    batch_spans2 = []

    for index, spans in enumerate(batch['span1']):
        for span in spans:
            batch_spans1.append((index, span[0], span[1]))
    for index, spans in enumerate(batch['span2']):
        for span in spans:
            batch_spans2.append((index, span[0], span[1]))
    return torch.tensor([batch_spans1, batch_spans2])

def train_loop(model, task_cfg, train, optimizer, criterion):
    '''
    Main training loop (for each epoch)
    '''
    data = train.sample(frac=task_cfg.train_size).reset_index(drop=True)
    model.train()
    stats = utils.StatTracker()
    for batch in tqdm(np.array_split(data, len(data) // task_cfg.batch_size), desc='Training'):
        optimizer.zero_grad()
        inputs = torch.tensor(np.vstack(batch['input_ids'])).to(device)
        if task_cfg.single_span:
            target_spans = get_span(batch).to(device)
        else:
            target_spans = get_span_double(batch).to(device)
        target_labels = torch.tensor(np.concatenate(batch['labels'].values)).to(device)
        outputs = model([inputs, target_spans])
        loss = criterion(outputs, target_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), task_cfg.gradient_clip)
        optimizer.step()
        outputs_softmax = torch.nn.functional.softmax(outputs, dim=1)
        acc = (outputs_softmax.argmax(1) == target_labels).sum().item()
        f1 = f1_score(target_labels.cpu().numpy(), outputs_softmax.argmax(1).cpu().numpy(), average='micro')
        stats.update(loss.item(), acc, len(target_labels), f1)
    return stats.get_stats()

def val_test_loop(model, task_cfg, data, criterion):
    '''
    Main validation and test loop
    '''
    model.eval()
    stats = utils.StatTracker()
    with torch.no_grad():
        for batch in tqdm(np.array_split(data, len(data) // task_cfg.batch_size), desc='Validation'):
            inputs = torch.tensor(np.vstack(batch['input_ids'])).to(device)
            if task_cfg.single_span:
                target_spans = get_span(batch).to(device)
            else:
                target_spans = get_span_double(batch).to(device)
            target_labels = torch.tensor(np.concatenate(batch['labels'].values)).to(device)
            outputs = model([inputs, target_spans])
            loss = criterion(outputs, target_labels)
            outputs_softmax = torch.nn.functional.softmax(outputs, dim=1)
            acc = (outputs_softmax.argmax(1) == target_labels).sum().item()
            f1 = f1_score(target_labels.cpu().numpy(), outputs_softmax.argmax(1).cpu().numpy(), average='micro')
            stats.update(loss.item(), acc, len(target_labels), f1)
    return stats.get_stats()

def save_model(model, save_path):
    '''
    Save the model to the path
    '''
    log.info(f'Saving model to {save_path}')
    save_file(model.state_dict(), save_path)

def train_task(model, task_cfg, wandb_logger):
    '''
    Train the model on one task
    '''
    train, val, test = load_data(task_cfg)
    optimizer, criterion, _ = training_modules(model, task_cfg)

    log.info(f'Starting training on task {task_cfg.name}...')
    for epoch in range(task_cfg.epochs):
        train_loss, train_acc, train_f1 = train_loop(model, task_cfg, train, optimizer, criterion)
        val_loss, val_acc, val_f1 = val_test_loop(model, task_cfg, data=val, criterion=criterion)
        log.info(f'{task_cfg.name} Epoch {epoch + 1}/{task_cfg.epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Train F1: {train_f1:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}')
        wandb_logger.log({'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1, 'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1})

    test_loss, test_acc, test_f1 = val_test_loop(model, task_cfg, data=test, criterion=criterion)
    log.info(f'{task_cfg.name} Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f} - Test F1: {test_f1:.4f}')
    wandb_logger.log({'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1})

@hydra.main(config_path='../config/', config_name='main')
def main(cfg):
    base_model = build_base_mdoel(cfg)
    wandb_logger = utils.WandbLogger(cfg)
    task_pooler = []
    task_probe = []

    # Train on each task sequentially
    for _, task_cfg in cfg.task.tasks.items():
        model = build_probing_model(base_model, task_cfg, cfg)
        train_task(model, task_cfg, wandb_logger)
        base_model = model.subject_model
        # Save the pooler and probe for each task to use in final testing
        task_pooler.append(model.pooler)
        task_probe.append(model.probe)

    # Test on all task after training
    log.info('Testing on all tasks...')
    for _, task_cfg in cfg.task.tasks.items():
        model = ProbingPair(base_model, task_pooler.pop(0), task_probe.pop(0), cfg.task.freeze_subject).to(device)
        _, _, test = load_data(task_cfg)
        _, criterion, _ = training_modules(model, task_cfg)
        log.info(f'Testing on task {task_cfg.name}...')
        test_loss, test_acc, test_f1 = val_test_loop(model, task_cfg, test, criterion)
        wandb_logger.log({'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1})
        log.info(f'{task_cfg.name} Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f} - Test F1: {test_f1:.4f}')

    log.info('Training finished!')
    if cfg.save_model:
        save_model(model, cfg.model.save_path + '.safetensor')

if __name__ == "__main__":
    main()