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
