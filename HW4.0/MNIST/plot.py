import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
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



datagen = ImageDataGenerator(
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')


from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
1
activation_model = models.Model(inputs=model.input,
outputs=layer_outputs) 2


plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
# 画图



def generate_pattern(layer_name, filter_index, size=150):
layer_output = model.get_layer(layer_name).output
1
loss = K.mean(layer_output[:, :, :, filter_index])
1
grads = K.gradients(loss, model.input)[0]
2
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
3
iterate = K.function([model.input], [loss, grads])
4
input_img_data = np.random.random((1, size, size, 3)) * 20 +
128. 5
step = 1.
6
for i in range(40):
6
loss_value, grads_value = iterate([input_img_data])
6
input_img_data += grads_value * step
6
img = input_img_data[0]
return deprocess_image(img)