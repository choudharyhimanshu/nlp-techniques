# @author : Himanshu Choudhary
# @home : http://www.himanshuchoudhary.com
# @git : https://bitbucket.org/himanshuchoudhary/

from __future__ import print_function
import numpy as np
from subprocess import call, check_output

from Utils import getBrownDatasetSentences

from sklearn.cross_validation import train_test_split

def zipTokensTags(tokens,tags):
    data = []
    for i in range(len(tokens)):
        data.append(tokens[i]+'_'+tags[i])
    return data

def prepareForOpenNLP(sentences,tags):
    data = []
    for i in range(len(sentences)):
        data.append(' '.join(zipTokensTags(sentences[i],tags[i])))
    return data

np.random.seed(1)

FILE_TRAIN = open('output/p1_e.dataset.train','w')
FILE_TEST = open('output/p1_e.dataset.test','w')
FILE_OUT = open('output/p1_e.txt','a')

dataset = getBrownDatasetSentences()

sentences = dataset['sentence'].values
tags = dataset['tags'].values

opennlp_dataset = prepareForOpenNLP(sentences,tags)

train_data, test_data = train_test_split(opennlp_dataset,test_size=.2)

for sentence in train_data:
    print(sentence,file=FILE_TRAIN)

for sentence in test_data:
    print(sentence,file=FILE_TEST)

call(['./lib/opennlp/bin/opennlp', 'POSTaggerTrainer', '-type', 'maxent', '-model', 'output/p1_e.trained.bin', \
      '-lang', 'en', '-data', 'output/p1_e.dataset.train', '-encoding', 'UTF-8'])

output = check_output(['./lib/opennlp/bin/opennlp', 'POSTaggerEvaluator', '-model', 'output/p1_e.trained.bin', \
      '-data', 'output/p1_e.dataset.test', '-encoding', 'UTF-8'])

needle = 'Accuracy: '
index = output.find(needle)

score = float(output[index+len(needle):])
print("Total sentences in the dataset : {:d}".format(len(dataset)), file=FILE_OUT)
print("# of sentences used as training set : {:d}".format(len(train_data)), file=FILE_OUT)
print("# of sentences used as testing set : {:d}".format(len(test_data)), file=FILE_OUT)
print("Accuracy : {:.2f}%".format(score*100), file=FILE_OUT)
print("\n", file=FILE_OUT)
