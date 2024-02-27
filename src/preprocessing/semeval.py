import pandas as pd
import logging as log

log.basicConfig(level=log.INFO)

class PreprocessSemeval():

    def __init__(self, path) -> None:
        
        self.path = path
        self.data = self._load_dataset()

    def _load_dataset(self):

        log.info('Loading SemEval dataset...')
        data = pd.read_parquet(self.path)
        log.info('Dataset shape: {}'.format(data.shape))
        return data
    
    def preprocess(self):

        log.info('Preprocessing SemEval dataset...')
        self.data['sentence'], self.data['e1_span'], self.data['e2_span'] = zip(*self.data['sentence'].apply(lambda x: self.process_sentence_v2(x)))

        log.info('Preprocessing completed.')
        log.info('Preprocessed dataset shape: {}'.format(self.data.shape))

        return self.data

    def process_sentence_v2(self, input_sentence):

        # Initialize variables to keep track of the modifications
        cleaned_sentence = ""
        tags = ['<e1>', '</e1>', '<e2>', '</e2>']
        e1_start, e1_end, e2_start, e2_end = [input_sentence.find(tag) for tag in tags]
        spans = []
        
        # Process the first entity
        if e1_start != -1 and e1_end != -1:
            spans.append((e1_start, e1_end - len('<e1>')))
        
        # Process the second entity
        if e2_start != -1 and e2_end != -1:
            # Adjust the start position of the second entity based on the removal of the first entity tags
            e2_start_adjusted = e2_start - len('<e1>') - len('</e1>')
            spans.append((e2_start_adjusted, e2_end - len('<e1>') - len('</e1>') - len('<e2>')))
        
        # Remove the tags to clean the sentence
        for tag in tags:
            input_sentence = input_sentence.replace(tag, ' ')
        
        cleaned_sentence = input_sentence.strip()

        # Adjust the end positions to match the required output format (exclusive end position)
        adjusted_spans = [(span[0], span[1] + 1) for span in spans]

        # Convert character spans to word spans
        word_spans = []
        for start_char, end_char in adjusted_spans:
            start_word = len(cleaned_sentence[:start_char].split())
            end_word = len(cleaned_sentence[:end_char].split())
            word_spans.append((start_word, end_word))

        return cleaned_sentence, word_spans[0], word_spans[1]

    def save(self, path):

        self.data.to_parquet(path, index=False)
        self.data.to_csv(path.replace('.parquet', '.csv'), index=False)
        log.info('Preprocessed dataset saved at {}'.format(path))
    
if __name__ == "__main__":

    split = 'train'
    path = f'../dataset/semeval_2010/semeval_2010_{split}.parquet'
    preprocessor = PreprocessSemeval(path)
    preprocessor.preprocess()
    preprocessor.save(f'../dataset/semeval_2010/semeval_2010_{split}_preprocessed.parquet')