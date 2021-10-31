import keras
import numpy as np
from keras import models
from keras import layers

from keras.layers import Flatten
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import pickle
from sklearn.model_selection import train_test_split


with open('Data.txt', 'rb') as fp:
    Vectorized = pickle.load(fp)
with open('Label.txt', 'rb') as fp:
    label = pickle.load(fp)
print(len(Vectorized))
print(len(label))

X_train, X_test, y_train, y_test = train_test_split(
    Vectorized, label, test_size=0.2, random_state=42)

model = models.Sequential()
model.add(layers.Conv1D(filters=1, kernel_size=4, activation='relu', strides=1,
          input_shape=(50, 1559)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Dense(16, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(l=0.02)))
model.add(layers.Conv1D(filters=16, kernel_size=2, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=8, kernel_size=2, activation='relu'))
# model.add(layers.Dense(1))
model.add(Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# print(len(X_train), len(y_train), len(X_test), len(y_test)) # 112 28
# print(type(X_train), type(y_train), type(X_test), type(y_test))
history = model.fit(X_train, y_train, epochs=40, verbose=2,
                    validation_data=(X_test, y_test))
history_dict = history.history
print('history_dict', history_dict.keys())

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

model.save("my_model")


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
