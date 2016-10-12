# @author : Himanshu Choudhary
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

from __future__ import print_function
import numpy as np

from Utils import getBrownDatasetAsItIs

from sklearn.cross_validation import train_test_split

from nltk.tag import str2tuple
from nltk.tag import UnigramTagger

def prepareForNLTK(sentence):
    return [str2tuple(word) for word in sentence.lower().split()]

np.random.seed(1)

FILE_OUT = open("output/p1_d.txt", 'a')

data = getBrownDatasetAsItIs().sentence.map(prepareForNLTK)
dataset = []
for x in data:
    dataset.append(x)

train_data, test_data = train_test_split(dataset,test_size=.2)

nltk_tagger = UnigramTagger(train_data)

score = nltk_tagger.evaluate(test_data)
print("Total sentences in the dataset : {:d}".format(len(dataset)), file=FILE_OUT)
print("# of sentences used as training set : {:d}".format(len(train_data)), file=FILE_OUT)
print("# of sentences used as testing set : {:d}".format(len(test_data)), file=FILE_OUT)
print("Accuracy : {:.2f}%".format(score*100), file=FILE_OUT)
print("\n", file=FILE_OUT)
