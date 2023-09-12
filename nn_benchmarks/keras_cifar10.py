# -*- coding: utf-8 -*-
"""keras_cifar10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dZkXw1cSOWTr2pkQ6go59CsxjsSuRFUo
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.models import Model

import numpy as np
import matplotlib.pyplot as plt

(Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.cifar10.load_data()

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(height_shift_range=0.2, width_shift_range=0.2, horizontal_flip=True, shear_range=0.1)

Ytrain = Ytrain.flatten()
Ytest = Ytest.flatten()

N = Xtrain.shape[0]
K = len(set(Ytrain) | set(Ytest))
batch_size = 32

Xtrain.shape

i = Input(shape=(32, 32, 3), batch_size=batch_size)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D((2, 2), 2)(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D((2, 2), 2)(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D((2, 2), 2)(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

train_generator = data_generator.flow(Xtrain, Ytrain, batch_size=batch_size)

steps_per_epoch = N // batch_size

r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=30, steps_per_epoch=steps_per_epoch)

r = model.fit(train_generator, validation_data=(Xtest, Ytest), epochs=30, steps_per_epoch=steps_per_epoch)

r = model.history

loss = r.history['loss']
val_loss = r.history['val_loss']
acc = r.history['accuracy']
val_acc = r.history['val_accuracy']

plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.legend()
plt.show()

plt.plot(acc, label='acc')
plt.plot(val_acc, label='val_acc')
plt.legend()
plt.show()