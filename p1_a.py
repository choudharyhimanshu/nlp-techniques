# @author : Himanshu Choudhary
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

import pandas as pd
from numpy import random
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from Utils import getBrownDatasetTokensWithTags

random.seed(1)

FILE_OUT = open("output/p1_a.txt", 'w')

def mostFrequentTag(tokens,tags,find_token):
    matched_tokens = (tokens[tokens == find_token])
    matched_tags = tags.get(matched_tokens.index)
    return matched_tags.value_counts().head(1).index[0]

def tagTokensWithMostFreqTag(tokens,tags):
    tagged = []
    done_tokens = []
    unique_tokens = tokens.unique()
    for token in unique_tokens:
        tag = mostFrequentTag(tokens, tags, token)
        tagged.append([token, tag])
        done_tokens.append(token)
    return pd.DataFrame(tagged, columns=['token', 'tag'])

dataset = getBrownDatasetTokensWithTags()

tokens = dataset['token']
tags = dataset['tag']

tokens_train, tokens_test, tags_train, tags_test = train_test_split(tokens, tags, test_size=0.2)

tagged_tokens = tagTokensWithMostFreqTag(tokens_train, tags_train)

tags_pred = []
for token in tokens_test:
    tag = tagged_tokens[tagged_tokens['token'] == token]
    if tag.empty:
        tags_pred.append('NA')
    else:
        tags_pred.append(tag['tag'].values[0])

score = accuracy_score(tags_test, tags_pred)
print("Total (token,tag) pairs in the dataset : %d" % len(dataset), file=FILE_OUT)
print("# of (token,tag) pairs used as training set : %d" % len(tokens_train), file=FILE_OUT)
print("# of (token,tag) pairs used as testing set : %d" % len(tokens_test), file=FILE_OUT)
print("# of unique tokens in the training set : %d" % len(tokens_train.unique()), file=FILE_OUT)
print("Accuracy : %f" % score, file=FILE_OUT)
