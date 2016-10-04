# @author : Himanshu Choudhary
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

from __future__ import print_function
import pandas as pd
from numpy import random
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from Utils import getIMDBTrainDataset, getIMDBTestDataset

from sklearn.feature_extraction.text import CountVectorizer

random.seed(1)

FILE_OUT = open("output/p1_a.txt", 'a')

DATASET_SIZE = 1000

dataset_train = getIMDBTrainDataset(.8*DATASET_SIZE)
dataset_test = getIMDBTestDataset(.2*DATASET_SIZE)

print(dataset_train)
print(dataset_test)
