# -*- coding: utf8 -*-
# build in libraries
import random

# tensorflow and keras
import tensorflow as tf
from tensorflow import keras

# helpers libraries
import matplotlib.pyplot as plt
import numpy as np

# local helper functions
from .helper import flatten_list, wrapper_func

print("tensorflow version: {version}".format(version=tf.__version__))

# fashion_mnist classification test
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# desc data shapes
print("train_image_shape: {shape},".format(shape=train_images.shape),)
print("train_label_shape: {shape}".format(shape=train_labels.shape))
print("test_image_shape: {shape}".format(shape=test_images.shape),)
print("test_label_shape: {shape}".format(shape=test_labels.shape))

# label decode
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# normalized input image
train_images_normalized = train_images / 255.0
test_images_normalized = test_images / 255.0

# show some image
fig, axes = plt.subplots(5, 5)
flatten_axes = flatten_list(axes.tolist())
images = [ flatten_axes[i].imshow(train_images[i]) for i in range(len(flatten_axes))]
plt.show()

# make model
model = keras.Sequential(
    layers=[
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(units=128, activation=tf.nn.relu),
        keras.layers.Dense(units=10, activation=tf.nn.softmax)
    ],
    name="MyFirstClsModel"
)

# make machine learning params
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# train_model
model.fit(x=train_images_normalized, y=train_labels, epochs=5)

# evaluate model
test_loss, test_acc = model.evaluate(x=test_images_normalized, y=test_labels)
print("test loss: {loss}".format(loss=test_loss),)
print("test acc: {acc}".format(acc=test_acc))

# predict one random image
predict_image_index = random.choice(range(len(test_images_normalized)))
predict_image = test_images_normalized[predict_image_index]
test_image_true_label_index = test_labels[predict_image_index]
test_image_true_label = labels[test_image_true_label_index]
pack_image = np.expand_dims(predict_image, 0)
predict_ans = model.predict(pack_image)
predict_ansuse = predict_ans[0]
predict_num = np.max(predict_ansuse) * 100
label_name_index = np.argmax(predict_ansuse)
predict_label_name = labels[label_name_index]

# print answer
print("{} {:2.0f}% {}".format(test_image_true_label, predict_num, predict_label_name))

# show predict answer
axes = plt.subplot(111)
axes.imshow(predict_image)
axes.set_xticks([])
axes.set_yticks([])
axes.set_xlabel("{} {:2.0f}% {}".format(test_image_true_label, predict_num, predict_label_name))
plt.show()
