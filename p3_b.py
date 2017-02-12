# @author : Himanshu Choudhary
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

from __future__ import print_function
from numpy import random
from Utils import getIMDBTrainDataset, getIMDBTestDataset

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

random.seed(1)

FILE_OUT = open("output/p3_b.txt", 'a')

DATASET_SIZE = 1000

dataset_train = getIMDBTrainDataset(.8*DATASET_SIZE)
dataset_test = getIMDBTestDataset(.2*DATASET_SIZE)

vectorizer = TfidfVectorizer(max_df=.8)
vectorizer.fit(dataset_train.review.values)

reviews_train = vectorizer.transform(dataset_train.review)
reviews_test = vectorizer.transform(dataset_test.review)

classifier = LinearSVC(verbose=1)
classifier.fit(reviews_train,dataset_train.polarity)

score = classifier.score(reviews_test,dataset_test.polarity)
print("Total sentences in the dataset : {:d}".format(DATASET_SIZE), file=FILE_OUT)
print("# of sentences used as training set : {:d}".format(len(dataset_train)), file=FILE_OUT)
print("# of sentences used as testing set : {:d}".format(len(dataset_test)), file=FILE_OUT)
print("Accuracy : {:.2f}%".format(score*100), file=FILE_OUT)
print("\n", file=FILE_OUT)
