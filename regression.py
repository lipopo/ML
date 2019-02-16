# -*- coding: utf8 -*-
# builtin model
import pathlib

# thired part model
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow import keras
import tensorflow as tf

# local model

# print tensorflow's version
print("tensorlfow version: {}".format(tf.__version__))

# load_data
data_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(
    data_path, names=column_names, na_values="?", comment="\t",
    sep=" ",skipinitialspace=True
)
dataset = raw_dataset.copy()
print("{v} dataset tail {v}".format(v="-"*20))
print(dataset.tail())
print("{v} dataset desc {v}".format(v="-"*20))
print(dataset.describe())

# calc unkown value
print("{v} nan value desc {v}".format(v="-"*20))
print(dataset.isna().sum())

# clean data
# drop nan value
dataset = dataset.dropna()

# put origin out
origin = dataset.pop("Origin")

# normalized dataset
dataset["USA"] = (origin==1) * 1.0
dataset["Europe"] = (origin==2) * 1.0
dataset["Japan"] = (origin==3) * 1.0

print("{v} clean data tail {v}".format(v="-"*20))
print(dataset.tail())

# split train and test data
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# inspect data
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

# split labels
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")

train_state = train_dataset.describe().T
print("{v} train_state {v}".format(v="-"*20))
print(train_state)

# normalized fit dataset
def norm(x):
    return (x - train_state["mean"]) / train_state["std"]

normed_train_dataset = norm(train_dataset)
normed_test_dataset = norm(test_dataset)

# build model
model = keras.Sequential(
    [
        keras.layers.Dense(units=64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        keras.layers.Dense(units=64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ],
    name="regression_model"
)


# set up compile params
model.compile(
    optimizer=keras.optimizers.RMSprop(0.001),
    loss="mse",
    metrics=[
        "mae",
        "mse"
    ]
)

# inspect model
model.summary()

# create monitor
monitor_val_loss = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

# fit model
history = model.fit(
    x=train_dataset,
    y=train_labels,
    epochs=1000,
    validation_split=0.2,
    verbose=0
)

# plot history
mae = history.history["mean_absolute_error"]
val_mae = history.history["val_mean_absolute_error"]
mse = history.history["mean_squared_error"]
val_mse = history.history["val_mean_squared_error"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(mae) + 1)

figure, axes = plt.subplots(3)
axes[0].plot(epochs, mae, "b", label="mae")
axes[0].plot(epochs, val_mae, "y", label="val_mae")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Mae")
axes[0].set_title("Mae and Vaildation Mae")
axes[0].set_ylim([0, 5])

axes[1].plot(epochs, mse, "b", label="mse")
axes[1].plot(epochs, val_mse, "y", label="val_mse")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Mse")
axes[1].set_title("Mse and Vaildation Mse")
axes[1].set_ylim([0, 20])

axes[2].plot(epochs, loss, "b", label="loss")
axes[2].plot(epochs, val_loss, "y", label="val_loss")
axes[2].set_xlabel("Epochs")
axes[2].set_ylabel("Loss")
axes[2].set_title("Loss and Vaildation Loss")

plt.legend()
plt.show()
