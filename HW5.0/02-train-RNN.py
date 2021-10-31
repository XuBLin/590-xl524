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


with open('Data2.txt', 'rb') as fp:
    Vectorized = pickle.load(fp)
with open('label2.txt', 'rb') as fp:
    label = pickle.load(fp)
# print(len(Vectorized))
# print(len(label))

X_train, X_test, y_train, y_test = train_test_split(
    Vectorized, label, test_size=0.2, random_state=42)
embedding_size = 20

# print(type(X_train[0][0]), type(X_test[0]), type(y_train[0]), type(y_test))
# print(X_test[0])

model = models.Sequential()

model.add(Embedding(3577, embedding_size, input_length=50))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(100))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, verbose=2,
                    validation_data=(X_test, y_test))
history_dict = history.history
print('history_dict', history_dict.keys())

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

results = model.evaluate(X_test, y_test)
print(results)
model.save("my_model_RNN")


epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss_values, 'r', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
