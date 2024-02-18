import pandas as pd
import logging as log

log.basicConfig(level=log.INFO)

class PreprocessOntonotes():

    def __init__(self, path):

        self.path = path

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

        log.info('Preprocessing completed.')
        log.info('Preprocessed dataset shape: {}'.format(df.shape))
        return df
    
    def save(self, path, df):

        log.info('Saving preprocessed Ontonotes dataset...')
        df.to_csv(path)
        log.info('Dataset saved at {}'.format(path))
        return
    
if __name__ == '__main__':

    path = '../dataset/ontonotesv5_english_v12_train.parquet'
    save_path = '../dataset/ontonotesv5_english_v12_train_processed.csv'
    preprocess = PreprocessOntonotes(path)
    df = preprocess.preprocess()
    preprocess.save(save_path, df)
    print(df.head())
