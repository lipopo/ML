# -*- coding: utf8 -*-
# builtin module
import random

# thried part module
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# local module

print("tensorflow version: {}".format(tf.__version__))

# load data
imdb = keras.datasets.imdb
(train_datas, train_labels), (test_datas, test_labels) = imdb.load_data(num_words=10000)

# desc dataset
print("train_datas shape: {}".format(train_datas.shape),)
print("train_labels shape: {}".format(train_labels.shape))
print("test_datas shape: {}".format(test_datas.shape),)
print("test_labels shape: {}".format(test_labels.shape))

# get word index
word_index = imdb.get_word_index()
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# normalized data
train_datas = keras.preprocessing.sequence.pad_sequences(
    train_datas,
    maxlen=256,
    value=word_index["<PAD>"],
    padding="post"
)

test_datas = keras.preprocessing.sequence.pad_sequences(
    test_datas,
    maxlen=256,
    value=word_index["<PAD>"],
    padding="post"
)

# deal work index
reverse_word_index = dict([(value, key) for key, value in word_index.items()])

def decode_review(text):
    return " ".join([reverse_word_index.get(i, '?') for i in text])

# decode test
data_choose = random.choice(train_datas)
print("origin data: %s" % " ".join(["{}".format(data_index) for data_index in data_choose]))
print("word decode: %s" % decode_review(data_choose))

# build model
vocab_size = 10000
model = keras.Sequential(
    [
        keras.layers.Embedding(vocab_size, 16),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(units=16, activation=tf.nn.relu),
        keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
    ],
    name="decode word"
)

model.summary()

# compile param
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss="binary_crossentropy",
    metrics=[
        "accuracy"
    ]
)

# make vaild datasset
x_val = train_datas[:10000]
partial_x_train = train_datas[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# train_model
history = model.fit(
    partial_x_train,
    partial_y_train,
    batch_size=512,
    epochs=40,
    validation_data=(x_val, y_val),
    verbose=1
)

# evaluate model
results = model.evaluate(x=test_datas, y=test_labels)
print("evaluate loss: {}, evaluate accuracy: {}".format(*results))

# plot fit status
acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)
axes = plt.subplot(211)
axes.plot(epochs, loss, "bo", label="Training loss")
axes.plot(epochs, val_loss, "b", label="Validation loss")
axes.set_title("Training and Validation loss")
axes.set_xlabel("Epochs")
axes.set_ylabel("Loss")

axes_2 = plt.subplot(212)
axes_2.plot(epochs, acc, "bo", label="Training acc")
axes_2.plot(epochs, val_acc, "b", label="Validation acc")
axes_2.set_title("Training and Validation acc")
axes_2.set_xlabel("Epochs")
axes_2.set_ylabel("Acc")

plt.legend()
plt.show()
