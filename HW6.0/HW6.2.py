from keras.datasets import mnist
from keras.datasets import fashion_mnist
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from random import randint

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
(train_images_f, train_labels_f), (test_images_f, test_labels_f) =\
    fashion_mnist.load_data()
train_images = train_images/np.max(train_images)
train_images_f = train_images_f/np.max(train_images_f)

train_images = train_images.reshape(60000, 28, 28, 1)
fashion_images = train_images_f.reshape(60000, 28, 28, 1)

model = models.Sequential()
model.add(layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu',
          padding='same', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
          padding="valid"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filters=48, kernel_size=(3, 3), activation='relu',
          padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='sigmoid'))

model.add(layers.Dense(432, activation='sigmoid'))
model.add(layers.Reshape((3, 3, 48)))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2DTranspose(16, (3, 3), activation='relu',
          padding="same"))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2DTranspose(8, (3, 3), activation='relu',
          padding="valid"))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2DTranspose(1, (3, 3), activation='relu',
          padding="same"))
model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.summary()

history = model.fit(train_images, train_images, epochs=30, batch_size=1000,
                    validation_split=0.2)
history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']


results = model.evaluate(train_images, train_images)
print('The loss of train MNIST is:', results)
results_f = model.evaluate(train_images, train_images)
print('The loss of train MNIST_Fashion is:', results_f)

epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'r', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss of MNIST')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save("HW6.1-model")
plt.clf()

for i in range(8):
    int = randint(0, 59999)
    plt.subplot(4, 4, i*2+1)
    plt.matshow(train_images[int].reshape(28, 28), fignum=False)
    output = model.predict(train_images[int].reshape(1, 28, 28, 1))
    plt.subplot(4, 4, i*2+2)
    plt.matshow(output.reshape(28, 28), fignum=False)

plt.show()
plt.clf()

for i in range(8):
    int = randint(0, 59999)
    plt.subplot(4, 4, i*2+1)
    plt.matshow(train_images_f[int].reshape(28, 28), fignum=False)
    output = model.predict(train_images_f[int].reshape(1, 28, 28, 1))
    plt.subplot(4, 4, i*2+2)
    plt.matshow(output.reshape(28, 28), fignum=False)

plt.show()
