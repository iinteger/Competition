import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler


tf.random.set_seed(0)
np.random.seed(0)

X_train = np.load("data/train_dataset.npy")
y_train = np.load("data/train_dataset_label.npy")
X_valid = np.load("data/validation_dataset.npy")
y_valid = np.load("data/validation_dataset_label.npy")
X_test = np.load("data/test_dataset.npy")
X_test = np.expand_dims(X_test, axis=-1)

y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

# modeling
nets = 25
model = [0] *nets
for i in range(nets):
    model[i] = Sequential()

    model[i].add(Conv2D(32, kernel_size=3, activation=tf.keras.activations.swish, input_shape=(28, 28, 1)))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(32, kernel_size=3, activation=tf.keras.activations.swish))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation=tf.keras.activations.swish))
    model[i].add(BatchNormalization())
    model[i].add(Dropout(0.4))

    model[i].add(Conv2D(64, kernel_size=3, activation=tf.keras.activations.swish))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(64, kernel_size=3, activation=tf.keras.activations.swish))
    model[i].add(BatchNormalization())
    model[i].add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation=tf.keras.activations.swish))
    model[i].add(BatchNormalization())
    model[i].add(Dropout(0.4))

    model[i].add(Conv2D(256, kernel_size=4, activation=tf.keras.activations.swish))
    model[i].add(BatchNormalization())
    model[i].add(Flatten())
    model[i].add(Dropout(0.4))
    model[i].add(Dense(10, activation='softmax'))

    model[i].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
# TRAIN NETWORKS
history = [0] * nets

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

datagen = ImageDataGenerator(
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15, # Randomly zoom image
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,
        shear_range=0.2,
        fill_mode="nearest")  # randomly shift images vertically (fraction of total height)

datagen.fit(X_train)

batch_size = 8
epochs = 100

for j in range(nets):
    history[j] = model[j].fit_generator(datagen.flow(X_train, y_train,
                                                     batch_size=batch_size),
                                        epochs=epochs,
                                        steps_per_epoch=X_train.shape[0]//batch_size,
                                        validation_data=(X_valid, y_valid),
                                        callbacks=[annealer], verbose=0)

results = np.zeros((X_test.shape[0], 10))

for j in range(nets):
    results = results + model[j].predict(X_test)
results = np.argmax(results, axis=1)

submission = pd.read_csv('./data/sample_submission.csv')
submission.iloc[:, 1] = np.array(results).astype(int)
submission.to_csv('submission_keras.csv', index=False)