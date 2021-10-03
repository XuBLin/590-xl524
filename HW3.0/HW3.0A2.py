import keras
from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# print(train_data[1])
# print("the length of comment2", len(train_data[1]))    # 189
# print('train comment1', train_labels[0])            # 1
# print("max index of word", max([max(sequence) for sequence in train_data]))  # 9999

# Translate
'''word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
print(decoded_review)'''

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# print(x_train.shape)       # (25000, 10000)
# print(x_test.shape)        # (25000, 10000)

# vectorize
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

'''Two intermediate layers with 16 hidden units each
A third layer that will output the scalar prediction regarding the
sentiment of the current review'''

model = models.Sequential()
# model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.05)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='mae', metrics=['accuracy'])

x_val = x_train[:5000]
partial_x_train = x_train[5000:]
y_val = y_train[:5000]
partial_y_train = y_train[5000:]

history = model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=334,
                    validation_data=(x_val, y_val))
history_dict = history.history

print('history_dict', history_dict.keys())
# history_dict dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
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

results = model.evaluate(x_test, y_test)

