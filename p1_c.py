# @author : Himanshu Choudhary
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from Utils import getBrownDatasetSentences

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, TimeDistributed

np.random.seed(1)

FILE_OUT = open("output/p1_c.txt", 'a')
MAX_SENT_LENGTH = 50

def indexizeToken(keys):
    return np.array([vocab.loc[key][0] for key in keys])

def encodeTag(index):
    if index == 0:
        index = 97  # TO BE FIXED
    vector = np.zeros(len(tagset),dtype=np.int32)
    vector[index-1] = 1
    return vector

def encodeTags(indices):
    return np.array([encodeTag(index) for index in indices])

def indexizeTag(keys):
    return np.array([tagset.loc[key][0] for key in keys])

dataset = getBrownDatasetSentences()

sentences = dataset['sentence']
sentences_tags = dataset['tags']

tokens = []
for sentence in sentences:
    tokens.extend(sentence)
vocab = list(set(tokens))
for i in range(len(vocab)):
    vocab[i] = [vocab[i],i]
vocab = pd.DataFrame(vocab,columns=['token','index']).set_index(['token'])

tags = []
for sentence_tag in sentences_tags:
    tags.extend(sentence_tag)
tagset = list(set(tags))
for i in range(len(tagset)):
    tagset[i] = [tagset[i],i+1]
tagset = pd.DataFrame(tagset,columns=['tag','index']).set_index(['tag'])

transformed_sentences = pad_sequences(sentences.map(indexizeToken),maxlen=MAX_SENT_LENGTH)
transformed_sentences_tags = np.array(map(encodeTags,pad_sequences(sentences_tags.map(indexizeTag),maxlen=MAX_SENT_LENGTH)))

X_train, X_test, y_train, y_test = train_test_split(transformed_sentences,transformed_sentences_tags,test_size=.2)

model = Sequential()
model.add(Embedding(len(vocab),128,input_length=MAX_SENT_LENGTH,dropout=0.2))
model.add(LSTM(len(tagset),input_dim=128,input_length=MAX_SENT_LENGTH,dropout_U=0.2,dropout_W=0.2,return_sequences=True))
model.add(TimeDistributed(Dense(len(tagset),init='normal',activation='softmax',input_shape=(128,MAX_SENT_LENGTH))))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, nb_epoch=20, batch_size=32)

score = model.evaluate(X_test, y_test)
print("Total sentences in the dataset : {:d}".format(len(dataset)), file=FILE_OUT)
print("# of sentences used as training set : {:d}".format(len(X_train)), file=FILE_OUT)
print("# of sentences used as testing set : {:d}".format(len(X_test)), file=FILE_OUT)
print("Vocab size : {:d}  Tagset size : {:d}".format(len(vocab),len(tagset)), file=FILE_OUT)
print("Accuracy : {:.2f}%".format(score[1]*100), file=FILE_OUT)
print("\n", file=FILE_OUT)
