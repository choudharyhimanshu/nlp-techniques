
import pandas as pd
import numpy as np
import codecs

from gensim.models import Word2Vec

class W2V:
    def __init__(self,vector_size,file_model,file_vocab):
        self.VECTOR_SIZE = vector_size
        self.FILE_MODEL = file_model
        self.FILE_VOCAB = file_vocab
        try:
            # self.model = Word2Vec.load(self.FILE_MODEL)
            self.model = Word2Vec.load_word2vec_format(self.FILE_MODEL,binary=False)
        except IOError:
            with codecs.open(self.FILE_VOCAB, 'r', encoding='utf-8') as file:
                vocab = file.read().splitlines()
                file.close()
            self.model = Word2Vec(size=self.VECTOR_SIZE, window=2, workers=1, min_count=1)
            self.model.build_vocab([vocab])
            self.save()

    def save(self):
        self.model.save(self.FILE_MODEL)
        # self.model.save_word2vec_format(self.FILE_MODEL,binary=False)

    def train(self,data):
        self.model.train(data)
        self.save()

    def transform(self,word):
        if word in self.model:
            return self.model[word]
        else:
            return np.zeros(self.VECTOR_SIZE)
