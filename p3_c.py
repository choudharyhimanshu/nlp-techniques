# @author : Himanshu Choudhary
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

from __future__ import print_function
import numpy as np
import pandas as pd
from Utils import getIMDBTrainDataset, getIMDBTestDataset

from nltk import word_tokenize
from gensim.models import Word2Vec

from sklearn.svm import LinearSVC

def tokenize(text):
    return word_tokenize(text.decode('utf-8'))

def averagedVectorize(data):
    return np.array([np.mean([vectorizer[token] for token in tokens if token in vectorizer] or [np.zeros(VECTOR_SIZE)], axis=0) for tokens in data])

np.random.seed(1)

FILE_OUT = open("output/p3_c.txt", 'a')

DATASET_SIZE = 3000
VECTOR_SIZE = 30

dataset_train = getIMDBTrainDataset(.8*DATASET_SIZE)
dataset_test = getIMDBTestDataset(.2*DATASET_SIZE)

reviews_train = dataset_train.review.map(tokenize)
reviews_test = dataset_test.review.map(tokenize)

vectorizer = Word2Vec(pd.concat([reviews_train,reviews_test]),size=VECTOR_SIZE,window=2,min_count=1,workers=1)

reviews_train = averagedVectorize(reviews_train.values)
reviews_test = averagedVectorize(reviews_test.values)

classifier = LinearSVC(verbose=1)
classifier.fit(reviews_train,dataset_train.polarity)

score = classifier.score(reviews_test,dataset_test.polarity)
print("Total sentences in the dataset : {:d}".format(DATASET_SIZE), file=FILE_OUT)
print("# of sentences used as training set : {:d}".format(len(dataset_train)), file=FILE_OUT)
print("# of sentences used as testing set : {:d}".format(len(dataset_test)), file=FILE_OUT)
print("Vector Size : {:d}".format(VECTOR_SIZE), file=FILE_OUT)
print("Accuracy : {:.2f}%".format(score*100), file=FILE_OUT)
print("\n", file=FILE_OUT)
