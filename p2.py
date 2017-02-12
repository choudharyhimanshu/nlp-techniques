# @author : Himanshu Choudhary
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

from __future__ import print_function
import numpy as np
import pandas as pd
import logging

from Utils import getBrownDatasetSentences

from sklearn.cross_validation import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from matplotlib import pyplot as plt

from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
np.random.seed(1)

FILE_OUT = open("output/p2.txt", 'a')
FILE_OUT_CSV = "output/p2.csv"
FILE_HIST_IMG = "output/p2-hist"
FILE_MODEL = "output/p2.word2vec.model"
FILE_1B_MODEL = "dataset/glove.6B.300d.txt"

def buildW2VModel(sentences):
    model = Word2Vec(sentences,size=300,window=5,min_count=1,workers=1)
    model.save_word2vec_format(FILE_MODEL,binary=False)

def loadW2VModel(file):
    return Word2Vec.load_word2vec_format(file,binary=False)

dataset = getBrownDatasetSentences().sentence.map(list)

# buildW2VModel(dataset)
# model = loadW2VModel(FILE_MODEL)
# model_1b = loadW2VModel(FILE_1B_MODEL)

# results = []
# for word in model.vocab:
#     if word in model_1b:
#         results.append([word,cosine_similarity([model[word]],[model_1b[word]])[0][0]])

# df = pd.DataFrame(results,columns=['word','similarity'])
# df.to_csv(FILE_OUT_CSV)

df = pd.read_csv(FILE_OUT_CSV,header=0,index_col=0)
similarity_scores = df['similarity']

plt.hist(similarity_scores,bins=10)
plt.xlabel("Cosine similarity")
plt.ylabel("Value count")
plt.savefig(FILE_HIST_IMG)
