# @author : Himanshu Choudhary
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

from __future__ import print_function
import numpy as np

from Utils import getBrownDatasetTokens

from sklearn.metrics import accuracy_score

from nltk import pos_tag

def tag(token):
    return pos_tag([token])[0][1].lower()

np.random.seed(1)

FILE_OUT = open("output/p1_d.txt", 'a')

dataset = getBrownDatasetTokens()

tokens = dataset.index
tags = dataset['tag'].values

nltk_tags = tokens.map(tag)

tagset = list(set(tags))
nltk_tagset = list(set(nltk_tags))

score = accuracy_score(tags, nltk_tags)
print("Total (token,tag) pairs in the dataset : {:d}".format(len(dataset)), file=FILE_OUT)
print("Tagset size : {:d}".format(len(tagset)), file=FILE_OUT)
print("NLTK tagset size : {:d}".format(len(nltk_tagset)), file=FILE_OUT)
print("Accuracy : {:.2f}%".format(score*100), file=FILE_OUT)
print("\n", file=FILE_OUT)
