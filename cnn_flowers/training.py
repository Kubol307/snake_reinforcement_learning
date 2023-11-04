import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import random


def classification_rate(Y, T):
    return Y == T

data_dir = os.path.join(os.getcwd(), 'dataset')

labels=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

dataset = []
for dir in os.listdir(data_dir):
    for img in os.listdir(os.path.join(data_dir, dir)):
        img_array = cv2.imread(os.path.join(data_dir,dir,img))
        img_array = cv2.resize(img_array, (200, 200))
        img_array = img_array/255.0
        # img_array = tf.keras.preprocessing.image.random_shear(img_array, intensity=0.2)
        # img_array = tf.keras.preprocessing.image.apply_channel_shift(img_array, intensity=0.4)
        dataset.append([img_array, labels.index(dir)])
    # print(dir)

# for i in range(100):
#     print(dataset[0][0][0][i])

# print(dataset[0])

plt.figure(figsize=(5, 5))
plt.imshow(dataset[0][0])   
plt.show() 

random.shuffle(dataset)

Xtrain = list(image[0] for image in dataset[:-int(0.4*len(dataset))])
Ytrain = list(image[1] for image in dataset[:-int(0.4*len(dataset))])
Xval = list(image[0] for image in dataset[-int(0.4*len(dataset)):-int(0.2*len(dataset))])
Yval = list(image[1] for image in dataset[-int(0.4*len(dataset)):-int(0.2*len(dataset))])
Xtest = list(image[0] for image in dataset[-int(0.2*len(dataset)):])
Ytest = list(image[1] for image in dataset[-int(0.2*len(dataset)):])


Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
Xval = np.array(Xval)
Yval = np.array(Yval)
Xtest = np.array(Xtest)
Ytest = np.array(Ytest)

# print(Ytest.shape)
# print(Ytest[1])

# print(Xtrain.shape)
# print(Xval.shape)
# print(set(Ytrain))
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, 
                                 (3, 3), 
                                 activation='relu', 
                                 input_shape=(200, 200, 3)))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Conv2D(256, 3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(512, 3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Dense(units=5))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                                  min_delta=0.01, 
                                                  patience=3, 
                                                  verbose=1)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='model/model_for_blob.h5', 
                                                      monitor='accuracy', 
                                                      verbose=1, 
                                                      save_best_only=True)

cb = [early_stopping, model_checkpoint]

print('starting training')
model.fit(Xtrain, Ytrain, batch_size=64, epochs=2, verbose=2, callbacks=cb, validation_data=(Xval, Yval))
print('finished')
# model.history

correct = []
# print(Xtest[0])
for i in range(Xtest.shape[0]):
    # print(Xtest[i])
    img_array = Xtest[i].reshape((1, 200, 200, 3))
    pred = np.argmax(model.predict(img_array))
    correct.append(classification_rate(pred, Ytest[i]))

print(f'classification rate: {np.mean(correct)}')

