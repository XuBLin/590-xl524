import keras
import numpy as np
from keras import models
from keras import layers

from keras.layers import Flatten, SpatialDropout1D
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import pickle
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, LSTM, Dense, Dropout


with open('Data.txt', 'rb') as fp:
    Vectorized = pickle.load(fp)
with open('Label.txt', 'rb') as fp:
    label = pickle.load(fp)

model = keras.models.load_model('my_model')

results = model.evaluate(Vectorized, label)
print('The loss and accuracy of Conv1D are:', results)

with open('Data2.txt', 'rb') as fp:
    Vectorized = pickle.load(fp)
with open('Label2.txt', 'rb') as fp:
    label = pickle.load(fp)

model = keras.models.load_model('my_model_RNN')

results = model.evaluate(Vectorized, label)
print('The loss and accuracy of LSTM are:', results)