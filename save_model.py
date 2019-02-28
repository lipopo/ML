# -*- coding: utf8 -*-
from __future__  import absolute_import, division, print_function
import os

import tensorflow as tf
from tensorflow import keras

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
BASE_DIR = os.path.dirname(os.path.relpath(__file__))

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# create model
def create_model():
    model = keras.Sequential(
        layers=[
            keras.layers.Dense(
                units=512, activation=tf.nn.relu, input_shape=(28 * 28,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(
                units=10, activation=tf.nn.softmax
            )
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=[
            "accuracy"
        ]
    )
    return model
model = create_model()
# desc model
model.summary()

# create check point
check_point_path = os.path.join(BASE_DIR, "assets/train_1/point.cpkt")
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=check_point_path,
    save_weights_only=True,
    verbose=1
)

# fit model
model.fit(
    train_images, train_labels, epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback]
)

# load from cpkt
new_model = create_model()
loss, acc = new_model.evaluate(test_images, test_labels)
print("New Model(before load weight): Loss {} Acc {}".format(loss, acc))

latest_cpkt = tf.train.latest_checkpoint(os.path.dirname(check_point_path))
new_model.load_weights(latest_cpkt)
loss, acc = new_model.evaluate(test_images, test_labels)
print("New Model(after load weight): Loss {} Acc {}".format(loss, acc))

# save weight 
weight_path = os.path.join(BASE_DIR, "assets/train_1/test_save")
model.save_weights(
    weight_path
)

new_model = create_model()
loss, acc = new_model.evaluate(test_images, test_labels)
print("New Model(before load weight): Loss {} Acc {}".format(loss, acc))

new_model.load_weights(weight_path)
loss, acc = new_model.evaluate(test_images, test_labels)
print("New Model(after load weight): Loss {} Acc {}".format(loss, acc))

# save model
model_path = os.path.join(BASE_DIR, "assets/train_1/model_saved")
model.save(model_path)

loss, acc = model.evaluate(test_images, test_labels)
print("Model: Loss {} Acc {}".format(loss, acc))

new_model = keras.models.load_model(model_path)
loss, acc = new_model.evaluate(test_images, test_labels)
print("New Model: Loss {} Acc {}".format(loss, acc))
