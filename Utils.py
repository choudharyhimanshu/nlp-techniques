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
        if 'cr' in file_name:
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
