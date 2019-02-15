# -*- coding: utf8 -*-
# builtin module

# thried part module
import tensorflow as tf
from tensorflow import keras

# local module

print("tensorflow version: {}".format(tf.__version__))

# load data
imdb = keras.datasets.imdb
(train_datas, train_labels), (test_datas, test_labels) = imdb.load_data(num_words=10000)

print("train_datas shape: {}".format(train_datas.shape),)
print("train_labels shape: {}".format(train_labels.shape))
print("test_datas shape: {}".format(test_datas.shape),)
print("test_labels shape: {}".format(test_labels.shape))
