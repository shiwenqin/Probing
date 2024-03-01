import wandb
from omegaconf import OmegaConf

class WandbLogger:
    def __init__(self, cfg):
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=config_dict,
        )
    
    def log(self, metrics):
        wandb.log(metrics)

class StatTracker:
    # Track and Calculate statistics
    def __init__(self):
        self.loss = 0
        self.acc = 0
        self.num = 0
        self.f1 = 0
        self.batch_num = 0

    def update(self, loss, acc, num, f1):
        self.loss += loss
        self.acc += acc
        self.num += num
        self.f1 += f1   
        self.batch_num += 1

    def get_stats(self):
        loss = self.loss / self.num
        acc = self.acc / self.num
        f1 = self.f1 / self.batch_num
        return loss, acc, f1

class EarlyStopper:
    # Standard early stopper
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def encode_batch(batch, tokenizer):
    """
    :param batch: a pandas DataFrame with columns 'sentence' and 'target_spans'.
    :param tokenizer: a tokenizer object from the HuggingFace's transformers library.
    """
    inputs = tokenizer(batch['sentence'].tolist(), return_tensors='pt', padding='longest', add_special_tokens=False)
    return inputs

def get_word_spans(sentence_batch, tokenizer):
    word_spans = []
    for batch_index, sentence in enumerate(sentence_batch):
        index = 0
        for word in sentence.split():
            tokenized_word = tokenizer.tokenize(word)
            word_spans.append((batch_index, index, index + len(tokenized_word)))
            index += len(tokenized_word)
    return word_spans

def get_spans_semeval(sentence, span1, span2, tokenizer, batch_index):
    res_span1 = [batch_index,0,0]
    res_span2 = [batch_index,0,0]
    current_index = 0
    for index, word in enumerate(sentence.split()):
        if index == span1[0]:
            res_span1[1] = current_index
        if index == span1[1]:
            res_span1[2] = current_index
        if index == span2[0]:
            res_span2[1] = current_index
        if index == span2[1]:
            res_span2[2] = current_index
        tokenized_word = tokenizer.tokenize(word)
        current_index += len(tokenized_word)
    return res_span1, res_span2

def get_spans_semeval_batch(batch, tokenizer):
    batch.reset_index(drop=True, inplace=True)
    batch_spans_1 = []
    batch_spans_2 = []
    for index, row in batch.iterrows():
        span1, span2 = get_spans_semeval(row['sentence'], row['e1_span'], row['e2_span'], tokenizer, index)
        batch_spans_1.append(span1)
        batch_spans_2.append(span2)
    return [batch_spans_1, batch_spans_2]
