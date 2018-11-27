import torch.utils.data as data
from gensim import Word2Vec
import numpy as np

class NLPDataset(data.Dataset):
    '''
    NLP Dataset
    '''
    def __init__(self, phrases, labels):
        self.wv = Word2Vec.load('./data/w2v_wiki300').wv
        X_ = []
        y_ = []
        for i, (phrase, label) in enumerate(zip(phrases, labels)):
            words = phrase.split(' ')
            try:
                vec = sum([self.wv[w] for w in words])
            except KeyError:
                continue
            X_.append(vec)
            y_.append(label)

        self.X = np.array(X_)
        self.y = y_

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

