# @author : Himanshu Choudhary
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

import numpy as np
import pandas as pd
from os import listdir

np.random.seed(1)

def getBrownDatasetAsItIs():
    files_list = listdir("dataset/brown/")
    data = []
    for file_name in files_list:
        if 'cr' in file_name or 'cc' in file_name:
            pass
        else:
            continue
        file = open("dataset/brown/" + file_name, 'r')
        sentences = [x.strip() for x in file.readlines() if x.strip()]
        data.extend(sentences)
    return pd.DataFrame(data,columns=['sentence'])

def getBrownDatasetSentences():
    files_list = listdir("dataset/brown/")
    data = []
    for file_name in files_list:
        if 'cr' in file_name or 'cc' in file_name:
            pass
        else:
            continue
        file = open("dataset/brown/" + file_name, 'r')
        sentences = [x.strip() for x in file.readlines() if x.strip()]
        for sentence in sentences:
            tokens = np.array([x.split('/') for x in sentence.lower().split() if len(x.split('/')) == 2])
            tags = tokens[:,1]
            tokens = tokens[:,0]
            data.append([tokens,tags])
    return pd.DataFrame(data,columns=['sentence','tags'])

def getBrownDatasetTokens():
    files_list = listdir("dataset/brown/")
    data = []
    for file_name in files_list:
        if 'cr' in file_name or 'cc' in file_name:
            pass
        else:
            continue
        file = open("dataset/brown/" + file_name, 'r')
        sentences = [x.strip() for x in file.readlines() if x.strip()]
        for sentence in sentences:
            tokens = [x.split('/') for x in sentence.lower().split() if len(x.split('/')) == 2]
            data.extend(tokens)
    return pd.DataFrame(data,columns=['token','tag']).set_index(['token'])

def getIMDBTrainDataset(limit):
    data = []
    count = 0
    files_list_neg = listdir("dataset/aclImdb/train/neg/")
    for file_name in files_list_neg:
        if count >= limit/2:
            break
        file = open("dataset/aclImdb/train/neg/" + file_name, 'r')
        review = file.readline().replace('<br />','')
        rating = file_name.split('_')[1].split('.')[0]
        data.append([review,rating])
        count += 1
    count = 0
    files_list_pos = listdir("dataset/aclImdb/train/pos/")
    for file_name in files_list_pos:
        if count >= limit/2:
            break
        file = open("dataset/aclImdb/train/pos/" + file_name, 'r')
        review = file.readline().replace('<br />','')
        rating = file_name.split('_')[1].split('.')[0]
        data.append([review, rating])
        count += 1
    np.random.shuffle(data)
    return pd.DataFrame(data,columns=['review','rating'])

def getIMDBTestDataset(limit):
    data = []
    count = 0
    files_list_neg = listdir("dataset/aclImdb/test/neg/")
    for file_name in files_list_neg:
        if count >= limit/2:
            break
        file = open("dataset/aclImdb/test/neg/" + file_name, 'r')
        review = file.readline().replace('<br />','')
        rating = file_name.split('_')[1].split('.')[0]
        data.append([review,rating])
        count += 1
    count = 0
    files_list_pos = listdir("dataset/aclImdb/test/pos/")
    for file_name in files_list_pos:
        if count >= limit/2:
            break
        file = open("dataset/aclImdb/test/pos/" + file_name, 'r')
        review = file.readline().replace('<br />','')
        rating = file_name.split('_')[1].split('.')[0]
        data.append([review, rating])
        count += 1
    np.random.shuffle(data)
    return pd.DataFrame(data,columns=['review','rating'])
