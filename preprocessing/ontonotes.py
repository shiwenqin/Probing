import pandas as pd
import logging as log
from random import randint

log.basicConfig(level=log.INFO)

class PreprocessOntonotes():

    def __init__(self, path, is_train=False):

        self.path = path
        self.is_train = is_train

        self.data = self._load_dataset()

    def _load_dataset(self):

        log.info('Loading Ontonotes dataset...')
        data = pd.read_parquet(self.path)
        log.info('Dataset shape: {}'.format(data.shape))
        return data
    
    def preprocess(self):

        log.info('Preprocessing Ontonotes dataset...')
        df = self.data.explode('sentences', ignore_index=True)
        df = pd.concat([df.drop(['sentences'], axis=1),
                        pd.DataFrame(df['sentences'].tolist())], 
                    axis=1)
        df = df[['document_id', 'words', 'pos_tags', 'named_entities']]
        df['sentence'] = df['words'].apply(lambda x: ' '.join(x))
        df = df.drop(['words'], axis=1)

        def _mapping_ner(ner):
            if ner == 0:
                return 0
            elif ner % 2 == 0:
                return int(ner / 2)
            else:
                return int(ner / 2) + 1
            
        def _mapping_ner_lst(lst):
            return [_mapping_ner(x) for x in lst]
        
        df.loc[:,'mapped_named_entities'] = df['named_entities'].apply(lambda x: _mapping_ner_lst(x))
        df = df.drop(['named_entities'], axis=1)

        df.loc[:,'pos_tags'] = df['pos_tags'].apply(list)

        if self.is_train:
            df = df.drop_duplicates(subset=['sentence'], keep='first')

        df = self._exclude_empty_pos(df)
        
        df = self._get_pos_label(df)

        log.info('Preprocessing completed.')
        log.info('Preprocessed dataset shape: {}'.format(df.shape))
        return df
    
    def _exclude_empty_pos(self, df):

        def _check_empty(pos_tags):
            percent_zeros = pos_tags.count(0) / len(pos_tags)
            return percent_zeros >= 0.6
        
        df = df[~df.pos_tags.apply(_check_empty)]
        return df
    
    def _get_pos_label(self, df):

        def _choose_pos(pos_tags):
            rand = randint(0, len(pos_tags)-1)
            return rand, pos_tags[rand]
        
        df.loc[:, 'pos_index'], df.loc[:, 'pos_label'] = zip(*df.pos_tags.apply(_choose_pos))

        return df
    
    def save(self, path, df):

        log.info('Saving preprocessed Ontonotes dataset...')
        df.to_csv(path, index=False)
        df.to_parquet(path.replace('.csv', '.parquet'), index=False)
        log.info('Dataset saved at {}'.format(path))
        return
    
if __name__ == '__main__':
    # test validation

    path = '../dataset/ontonotesv5_english_v12_validation.parquet'
    save_path = '../dataset/ontonotesv5_english_v12_validation_processed_excludepos.csv'
    preprocess = PreprocessOntonotes(path, is_train=False)
    df = preprocess.preprocess()
    preprocess.save(save_path, df)
    print(df.head())
