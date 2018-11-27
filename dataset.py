import torch.utils.data as data
from gensim import Word2Vec
import numpy as np

class NLP_Dataset(data.Dataset):
    '''
    NLP Dataset
    '''
    def __init__(self, words, labels):
        self.wv = word2Vec.load('./data/w2v_wiki300').wv
        for i, w in enumerate(words):
            if w not in wv:
                words.pop(i)
                labels.pop(i)
        self.X = wv[np.array(words)]
        self.y = np.array(labels)

    def __len__(self):
        return len(y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

