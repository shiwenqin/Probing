
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