
import numpy as np
import pandas as pd
from os import listdir
from word2vec import W2V

FILE_VOCAB = "dataset/aclImdb/imdb.vocab"
FILE_W2V_MODEL = "output/imdb_trained.word2vec.model"

DATASET_SIZE = 5000
OFFSET = 25000
VECTOR_SIZE = 40

def getIMDBTrainDataset(limit,offset):
    data = []
    count = 0
    files_list_neg = listdir("dataset/aclImdb/train/neg/")
    for file_name in files_list_neg:
        if count < offset/2:
            count += 1
            continue
        if count >= (offset+limit)/2:
            break
        file = open("dataset/aclImdb/train/neg/" + file_name, 'r')
        review = file.readline().replace('<br />','')
        rating = file_name.split('_')[1].split('.')[0]
        data.append([review,int(rating),0])
        count += 1
    count = 0
    files_list_pos = listdir("dataset/aclImdb/train/pos/")
    for file_name in files_list_pos:
        if count < offset/2:
            count += 1
            continue
        if count >= (offset+limit)/2:
            break
        file = open("dataset/aclImdb/train/pos/" + file_name, 'r')
        review = file.readline().replace('<br />','')
        rating = file_name.split('_')[1].split('.')[0]
        data.append([review, int(rating),1])
        count += 1
    np.random.shuffle(data)
    return pd.DataFrame(data,columns=['review','rating','polarity'])

dataset = getIMDBTrainDataset(DATASET_SIZE,OFFSET)

vectorizer = W2V(vector_size=VECTOR_SIZE,file_vocab=FILE_VOCAB,file_model=FILE_W2V_MODEL)

vectorizer.train(dataset.review)
