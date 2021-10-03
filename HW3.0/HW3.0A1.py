from keras.datasets import boston_housing
import keras
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
              input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.02)))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
# print(train_data.shape)   # (404, 13)
# print(test_data.shape)   # (102, 13)
print(train_data.shape)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

# num_epochs = 500
all_mae_histories = []
all_test_mse = []
all_test_mae = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()

    ########
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets), epochs=num_epochs,
                        batch_size=10, verbose=0)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
    all_test_mse.append(test_mse_score)
    all_test_mae.append(test_mae_score)
    # print(history.history)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

average_mae_history = []
for i in range(num_epochs):
    avg = np.mean([x[i] for x in all_mae_histories])
    average_mae_history.append(avg)
# print("average_mae_history", average_mae_history)


'''for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)'''


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


smooth_mae_history = smooth_curve(average_mae_history[10:])
print("min_mae", np.min(smooth_mae_history))
print("len", len(smooth_mae_history))
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history, 'r', label="smooth")
plt.plot(range(1, len(smooth_mae_history) + 1), average_mae_history[10:10+len(smooth_mae_history)],
         'b', label='original')

plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.legend()
plt.show()
print("all_test_mae", all_test_mae)
print("The mean of mae is", np.mean(all_test_mae))


model2 = build_model()
model2.fit(train_data, train_targets, epochs=90, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print("Final MAE score:", test_mae_score)

