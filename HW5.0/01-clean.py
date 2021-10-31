import keras
from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pickle


path = os.getcwd()
files = os.listdir(path+'/Books')
Corpus = []
length_of_strings = []
for file in files:
    txt = open(path+"/Books/"+file, encoding="utf8", errors='ignore')
    text = txt.read()
    txt.close()
    words = re.sub(r"[^A-Za-z\-\n']", " ", text)
    strings = words.split("\n\n")
    for i in range(len(strings)):
        strings[i] = re.sub('\n', ' ', strings[i])
    strings = [para for para in strings if len(para) > 10]
    # print(len(strings))  # 1829, 2201, 1430
    strings = strings[0:50]
    Corpus = Corpus + strings

label0 = [0] * 50 + [1] * 50 + [2] * 50
label = []

flat = []
for i in range(len(Corpus)):
    para_split = Corpus[i].split(" ")
    para_split = [word for word in para_split if len(word) > 0]
    if len(para_split) > 10:
        if len(para_split) > 50:
            length_of_strings.append(50)
            flat = flat + para_split[0:50]
        else:
            length_of_strings.append(len(para_split))
            flat = flat + para_split
        label.append(label0[i])

# print(length_of_strings)
maxlength = max(length_of_strings)
print(maxlength)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(flat)
print(X.shape)     # total words, vocabulary
Vectorized = []
head = 0
pad = [0]*X.shape[1]
for i in range(len(length_of_strings)):
    temp = X[head:head+length_of_strings[i]].toarray()
    pad2 = [pad]*(maxlength-length_of_strings[i])
    if maxlength != length_of_strings[i]:
        temp = np.append(temp, pad2, axis=0)
    Vectorized.append(temp.tolist())
    # print(temp.shape)  # 158words, 2040kinds of
    head = head + length_of_strings[i]

with open('Data.txt', 'wb') as fp:
    pickle.dump(Vectorized, fp)

with open('Label.txt', 'wb') as fp:
    pickle.dump(label, fp)
