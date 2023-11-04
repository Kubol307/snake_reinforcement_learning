import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

data_dir = os.path.join(os.getcwd(), 'dataset')
train_dir = os.path.join(data_dir)
# val_dir = os.path.join(data_dir, 'validation')
# test_dir = os.path.join(data_dir, 'test')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(shear_range=0.2, 
                                                                zoom_range=0.5, 
                                                                horizontal_flip=True, 
                                                                rescale=1./255, 
                                                                validation_split=0.15, 
                                                                preprocessing_function=preprocess_input)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(train_dir, target_size=(320, 240), batch_size=16)
val_dataset = train_datagen.flow_from_directory(train_dir, target_size=(320, 240), subset='validation', batch_size=16)
# test_dataset = test_datagen.flow_from_directory(test_dir, target_size=(320, 240))

train_images, train_labels = train_dataset.next()

def image_show(image):
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.show()

# image_show(train_images[-1])

base_model = VGG19(input_shape=(320, 240, 3), include_top=False)

for layer in base_model.layers:
    layer.trainable = False

X = tf.keras.layers.Flatten()(base_model.output)
X = tf.keras.layers.Dense(activation='softmax', units=5)(X)

model = tf.keras.models.Model(base_model.input, X)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.01, patience=3, verbose=1)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./model/model_for_blob', monitor='accuracy', verbose=1, save_best_only=True)

callback = [early_stopping, model_checkpoint]
# print(len(set(train_dataset.labels)))
model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.fit_generator(generator=train_dataset, 
          epochs=2, 
          verbose=1, 
          validation_data=val_dataset, 
          validation_steps=32,
          callbacks=callback,
          steps_per_epoch=16)

