# @author : Himanshu Choudhary
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

from __future__ import print_function
import numpy as np
import pandas as pd
from Utils import getIMDBTrainDataset, getIMDBTestDataset

from nltk import word_tokenize
from word2vec import W2V

from sklearn.svm import LinearSVC

# from keras.models import Sequential
# from keras.layers import TimeDistributed, Dense, LSTM

np.random.seed(1)

FILE_VOCAB = "dataset/aclImdb/imdb.vocab"
FILE_W2V_MODEL = "dataset/glove.6B.50d.txt"
# FILE_W2V_MODEL = "output/imdb_trained.word2vec.model"
FILE_OUT = open("output/p3_d.txt", 'a')

DATASET_SIZE = 3000
VECTOR_SIZE = 100
MAX_REVIEW_LEN = 100

def tokenize(text):
    return word_tokenize(text.decode('utf-8'))

def pad(tokens):
    return np.array(tokens[:MAX_REVIEW_LEN] if len(tokens)>MAX_REVIEW_LEN else
                        np.concatenate((tokens,['' for _ in range(MAX_REVIEW_LEN-len(tokens))])))

def vectorize(data):
    return np.array([np.array([vectorizer.transform(token) for token in tokens]) for tokens in data])

dataset_train = getIMDBTrainDataset(.8*DATASET_SIZE)
dataset_test = getIMDBTestDataset(.2*DATASET_SIZE)

reviews_train = dataset_train.review.map(tokenize)
reviews_test = dataset_test.review.map(tokenize)

# vectorizer = Word2Vec(pd.concat([reviews_train,reviews_test]),size=VECTOR_SIZE,window=2,min_count=1,workers=1)
# vectorizer.train(reviews_train)

vectorizer = W2V(vector_size=VECTOR_SIZE,file_vocab=FILE_VOCAB,file_model=FILE_W2V_MODEL)

reviews_train = reviews_train.map(pad)
reviews_test = reviews_test.map(pad)

reviews_train = vectorize(reviews_train.values)
reviews_test = vectorize(reviews_test.values)

classifier = LinearSVC(verbose=1)
classifier.fit(reviews_train.reshape(len(reviews_train),MAX_REVIEW_LEN*VECTOR_SIZE),dataset_train.polarity)
score = classifier.score(reviews_test.reshape((len(reviews_test),MAX_REVIEW_LEN*VECTOR_SIZE)),dataset_test.polarity)

# model = Sequential()
# model.add(LSTM(64,input_dim=VECTOR_SIZE,input_length=MAX_REVIEW_LEN,dropout_U=.2,dropout_W=.2))
# model.add(Dense(1,activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(reviews_train,dataset_train.polarity,nb_epoch=20,batch_size=32)
# score = model.evaluate(reviews_test,dataset_test.polarity)[1]

print("Total sentences in the dataset : {:d}".format(DATASET_SIZE), file=FILE_OUT)
print("# of sentences used as training set : {:d}".format(len(dataset_train)), file=FILE_OUT)
print("# of sentences used as testing set : {:d}".format(len(dataset_test)), file=FILE_OUT)
print("Vector Size : {:d}   Review Length : {:d}".format(VECTOR_SIZE,MAX_REVIEW_LEN), file=FILE_OUT)
print("Accuracy : {:.2f}%".format(score*100), file=FILE_OUT)
print("\n", file=FILE_OUT)
