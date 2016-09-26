# @author : Himanshu Choudhary
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

from __future__ import print_function
import pandas as pd
from numpy import random
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from Utils import getBrownDatasetTokensWithTags

random.seed(1)

FILE_OUT = open("output/p1_a.txt", 'a')

def tagTokensWithMostFreqTag(data):
    tagged = []
    unique_tokens = data.index.unique()
    for token in unique_tokens:
        matched_rows = data.loc[token]
        if matched_rows.size == 1:
            tag = matched_rows.tag
        else:
            tag = matched_rows.tag.value_counts().head(1).index[0]
        tagged.append([token, tag])
    return pd.DataFrame(tagged, columns=['token', 'tag']).set_index(['token'])

dataset = getBrownDatasetTokensWithTags()

dataset_train, dataset_test = train_test_split(dataset, test_size=0.2)

tagged_tokens = tagTokensWithMostFreqTag(dataset_train)

tags_pred = []
for token in dataset_test.index:
    try:
        tag = tagged_tokens.loc[token].tag
    except KeyError:
        tag = 'NA'
    tags_pred.append(tag)

score = accuracy_score(dataset_test.values, tags_pred)
print('Total (token,tag) pairs in the dataset : %d' % len(dataset), file=FILE_OUT)
print('# of (token,tag) pairs used as training set : %d' % len(dataset_train), file=FILE_OUT)
print('# of (token,tag) pairs used as testing set : %d' % len(dataset_test), file=FILE_OUT)
print('# of unique tokens in the training set : %d' % len(dataset_train.index.unique()), file=FILE_OUT)
print('Accuracy : %f' % score, file=FILE_OUT)
print("\n", file=FILE_OUT)
