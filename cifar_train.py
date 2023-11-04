import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np


(Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.cifar10.load_data()
print(len(set(Ytrain.flatten())))

Xtrain, Xtest = Xtrain/255.0 , Xtest/255.0

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, 3, input_shape=(32, 32, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# early_stopping = tf.keras.callbacks.EarlyStopping(min_delta=0.001, patience=3)
# mc = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', save_best_only=True)
# cb = [early_stopping, mc]
model.fit(Xtrain, Ytrain, batch_size=32, epochs=8)