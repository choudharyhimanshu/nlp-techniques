# @author : Himanshu Choudhary
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

import numpy as np
import pandas as pd
from os import listdir

def getBrownDatasetSentences():
    files_list = listdir("dataset/brown/")
    data = []
    for file_name in files_list:
        file = open("dataset/brown/" + file_name, 'r', errors='ignore')
        sentences = [x.strip() for x in file.readlines() if x.strip()]
        data.extend(sentences)
    return np.array(data)

def  getBrownDatasetTokens():
    files_list = listdir("dataset/brown/")
    data = []
    for file_name in files_list:
        file = open("dataset/brown/" + file_name, 'r', errors='ignore')
        sentences = [x.strip() for x in file.readlines() if x.strip()]
        for sentence in sentences:
            tokens = [x.split('/')[0] for x in sentence.split() if '//' not in x and len(x.split('/')) == 2]
            data.extend(tokens)
    return pd.DataFrame(data,columns=['token','tag'])

def getBrownDatasetTokensWithTags():
    files_list = listdir("dataset/brown/")
    data = []
    count=0
    for file_name in files_list:
        file = open("dataset/brown/" + file_name, 'r', errors='ignore')
        sentences = [x.strip() for x in file.readlines() if x.strip()]
        for sentence in sentences:
            tokens = [x.split('/') for x in sentence.split() if '//' not in x and len(x.split('/')) == 2]
            data.extend(tokens)
        if count > 5:
            break
        count+=1
    return pd.DataFrame(data,columns=['token','tag'])
