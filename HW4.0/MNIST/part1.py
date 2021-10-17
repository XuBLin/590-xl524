from keras import layers
from keras import models
from keras.datasets import mnist, cifar10
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import random
import keras
from keras.preprocessing.image import ImageDataGenerator


Flag = 'cifar10'
Bsize = 64
Epo = 5         # 100 for aug
Show_Image = False
Opt = 'rmsprop'
Loss = 'categorical_crossentropy'
Metric = ['accuracy']
Net = 'CNN'
Aug = False
model_save = False
model_load = False
model_route = 'my_model'
model_viz = True

trainshape_MNIST = (60000, 28, 28, 1)
testshape_MNIST = (10000, 28, 28, 1)
trainshape_cifar = (50000, 32, 32, 3)
testshape_cifar = (10000, 32, 32, 3)


def show_image(dataset, train_images):
    if dataset == 'MNIST' or dataset == 'MNIST Fashion':
        num_to_show = random.randint(0, train_images.shape[0]-1)
        img = train_images[num_to_show]
        img = np.array(img, dtype='float')
        plt.imshow(img, cmap='Greys', interpolation='nearest')
        plt.show()
    elif dataset == 'cifar10':
        num_to_show = random.randint(0, train_images.shape[0]-1)
        img = train_images[num_to_show]
        img = np.array(img, dtype='float')
        plt.imshow(img)
        plt.show()


def preprocess(dataset):
    global trainshape_MNIST, testshape_MNIST, trainshape_cifar, testshape_cifar
    if dataset == 'MNIST' or dataset == 'MNIST Fashion':
        trainshape = trainshape_MNIST
        testshape = testshape_MNIST
        (train_images, train_labels),\
            (test_images, test_labels) = mnist.load_data()
    elif dataset == 'cifar10':
        trainshape = trainshape_cifar
        testshape = testshape_cifar
        (train_images, train_labels),\
            (test_images, test_labels) = cifar10.load_data()

    train_images = train_images.reshape(trainshape)
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape(testshape)
    test_images = test_images.astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images, test_images, train_labels, test_labels, trainshape


def buildmodel(dataset):
    if not model_load:
        if dataset == 'MNIST' or dataset == 'MNIST Fashion':
            Inshape = (28, 28, 1)
        elif dataset == 'cifar10':
            Inshape = (32, 32, 3)
        if Net == 'CNN':
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=Inshape))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(10, activation='softmax'))
            model.summary()

            model.compile(optimizer=Opt, loss=Loss, metrics=Metric)
        elif Net == 'ANN':
            model = models.Sequential()
            model.add(layers.Dense(64, activation='relu', input_shape=Inshape))
            model.add(layers.Flatten())
            model.add(layers.Dense(32, activation='relu',
                      kernel_regularizer=keras.regularizers.l2(l=0.02)))
            model.add(layers.Dense(10, activation='softmax'))
            model.compile(optimizer=Opt, loss=Loss, metrics=Metric)
    else:
        model = keras.models.load_model(model_route)
    return model


train_images, test_images, train_labels,\
    test_labels, trainshape = preprocess(Flag)
if Show_Image:
    show_image(Flag, train_images)
model = buildmodel(Flag)
if Aug:
    datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,
                                 height_shift_range=0.2, shear_range=0.2,
                                 zoom_range=0.2, horizontal_flip=True,
                                 fill_mode='nearest', validation_split=0.2)
    train_images = np.array(train_images)
    train_images = train_images.reshape(trainshape_MNIST)
    datagen.fit(train_images)
    train_generator = datagen.flow(train_images, train_labels,
                                   batch_size=Bsize, subset='training')
    validation_generator = datagen.flow(train_images, train_labels,
                                        batch_size=Bsize, subset='validation')
    history = model.fit_generator(generator=train_generator,
                                  validation_data=validation_generator,
                                  use_multiprocessing=True,
                                  steps_per_epoch=len(train_generator)/60,
                                  validation_steps=len(validation_generator)/60,
                                  epochs=Epo, workers=-1)
else:
    history = model.fit(train_images, train_labels, epochs=Epo,
                        batch_size=Bsize, validation_split=0.2)


test_loss, test_acc = model.evaluate(test_images, test_labels)
if model_save:
    model.save("my_model")
print("The test accuracy is:", test_acc)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

if model_viz:
    num_to_show = random.randint(0, train_images.shape[0]-1)
    img_tensor = train_images[num_to_show]
    img_tensor = np.array(img_tensor, dtype='float')
    img_tensor = img_tensor.reshape((1,) + trainshape[1:])
    layer_outputs = [layer.output for layer in model.layers[:3]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    layer_names = []
    for layer in model.layers[:3]:
        layer_names.append(layer.name)

    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                             row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()
