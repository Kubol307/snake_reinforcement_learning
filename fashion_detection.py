import tensorflow as tf


(x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)

    return x, y

def create_dataset(xs, ys, n_classes=10):
    ys = tf.one_hot(ys, depth=n_classes)
    return tf.data.Dataset.from_tensor_slices((xs,ys))  
        # .map(preprocess) \
        # .shuffle(len(ys)) \ 
        # .batch(128)

train_dataset = create_dataset(x_train, y_train)
validation_dataset = create_dataset(x_val, y_val)

model = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=(784), input_shape=(28,28)),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=192, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

hisotry = model.fit(
    train_dataset.repeat(),
    epochs=10,
    steps_per_epoch=500,
    validation_data=validation_dataset.repeat(),
    validation_steps=2
)