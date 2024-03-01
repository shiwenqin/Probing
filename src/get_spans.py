import numpy as np

def choose_span_getter(cfg):
    if cfg.task.name == 'pos_tagging':
        return GetSpansPOSOntoNote()
    elif cfg.task.name == 'semeval':
        return GetSpansSemEval()
    else:
        raise ValueError(f'Invalid task name {cfg.task.name}')

class GetSpansPOSOntoNote():

    def __init__(self):
        super().__init__()

    def get_span_single(self, batch_index, sentence, tokenizer):
        """
        Get the span for a single example
        """
        pos_spans = []
        index = 0
        for word in sentence.split():
            tokenized_word = tokenizer.tokenize(word)
            pos_spans.append((batch_index, index, index + len(tokenized_word)))
            index += len(tokenized_word)
        return pos_spans

    def get_span_batch(self, batch, tokenizer):
        """
        Get the span for a batch of examples
        """
        pos_spans = []
        for index, row in enumerate(batch['sentence']):
            span = self.get_span_single(index, row, tokenizer)
            pos_spans += span
        #print(pos_spans[-1])
        return pos_spans
    
    def get_labels(self, batch):
        """
        Get the labels for a batch of examples
        """
        return np.concatenate(batch['pos_tags'].values)

class GetSpansSemEval():

    def __init__(self):
        super().__init__()

    def get_span_single(self, sentence, span1, span2, tokenizer, batch_index):
        """
        Get the span for a single example
        """
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

    def get_span_batch(self, batch, tokenizer):
        """
        Get the span for a batch of examples
        """
        batch.reset_index(drop=True, inplace=True)
        batch_spans_1 = []
        batch_spans_2 = []
        for index, row in batch.iterrows():
            span1, span2 = self.get_span_single(row['sentence'], row['e1_span'], row['e2_span'], tokenizer, index)
            batch_spans_1.append(span1)
            batch_spans_2.append(span2)
        return [batch_spans_1, batch_spans_2]
    
    def get_labels(self, batch):
        """
        Get the labels for a batch of examples
        """
        return batch['relation'].values