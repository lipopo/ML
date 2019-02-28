# -*- coding: utf8 -*-
# 自定义训练: 演示
import os

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib.eager as tfe


print(tf.__version__)
tf.enable_eager_execution()

# 步骤
# 1.导入和解析数据集
# 2.选择和构建模型
# 3.训练模型
# 4.评估模型效果
# 5.使用经过训练的模型进行预测

# 导入和解析数据
train_set_url = "http://download.tensorflow.org/data/iris_training.csv"
iris_fp = keras.utils.get_file(
    fname=os.path.basename(train_set_url),
    origin=train_set_url
)

test_url = "http://download.tensorflow.org/data/iris_test.csv"

iris_test_fp = keras.utils.get_file(
    fname=os.path.basename(test_url),
    origin=test_url)

iris_data = pd.read_csv(
    iris_fp, 
    header=None,
    skiprows=1, 
    names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    )
iris_data = iris_data
print(iris_data.head())
print(iris_data.describe())


class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
# 查看数据
sns.pairplot(
    iris_data, hue='species', 
    x_vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    y_vars=["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
plt.show()

# 生成训练数据集
batch_size = 32
train_set = tf.contrib.data.make_csv_dataset(
    iris_fp,
    batch_size,
    column_names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
    label_name="species",
    num_epochs=1
)

test_set = tf.contrib.data.make_csv_dataset(
    iris_test_fp,
    batch_size,
    column_names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
    label_name="species",
    num_epochs=1
)

# 测试生成数据
features, labels = next(iter(train_set))
print("features: \n{}\n".format(features))

plt.scatter(
    features["petal_length"],
    features["sepal_length"],
    c=labels,
    cmap="viridis"
)

plt.xlabel("petal_length")
plt.ylabel("sepal_length")
plt.show()

# 将生成的模型压制成合适的形状
def pack_features_vetor(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

train_dataset = train_set.map(pack_features_vetor)
test_dataset = test_set.map(pack_features_vetor)
features, labels = next(iter(train_dataset))

print("features: \n{}\n".format(features))
print("labels: \n{}\n".format(labels))

# 构建我们的模型
model = keras.Sequential(
    layers=[
        keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(None, 4)),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(3)
    ]
)

# 测试我们的模型
predictions = model(features)
prediction_activation = tf.nn.softmax(predictions)

print("Prediction: {}".format(tf.argmax(prediction_activation, axis=1)))
print("True labels: {}".format(labels))

# 训练我们的模型

# 定义一个损失函数
def loss(model, x, y):
    y_ = model(x)
    loss_ = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
    return loss_

# 计算一下目前的误差
l = loss(model, features, labels)
print("current loss: \n{}\n".format(l))

# 获取损失和梯度
def grad(model, inputs, labels):
    with tf.GradientTape() as t:
        loss_value = loss(model, inputs, labels)
    deleta_v = t.gradient(loss_value, model.trainable_variables)
    return loss_value, deleta_v

# 定义优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# 定义全局计数器
global_step = tf.train.get_or_create_global_step()

# 尝试训练一轮
loss_value, grad_value = grad(model, features, labels)
print("Step {} Initial Loss {}".format(global_step.numpy(), loss_value.numpy()))
# 优化模型
optimizer.apply_gradients(zip(grad_value, model.variables), global_step)
# 查看优化效果
print("Step {} Loss {}".format(global_step.numpy(), loss(model, features, labels)))

# 循环训练模型
train_loss_results = []
train_accuracy_results = []

num_epochs = 200

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    for x, y in train_dataset:
        # 计算梯度
        loss_value, grad_value = grad(model, x, y)
        optimizer.apply_gradients(zip(grad_value, model.variables), global_step)

        # 记录损失值
        epoch_loss_avg(loss_value)
        # 记录准确率
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {} Loss: {:.3f} Accuracy: {:.3%}".format(
            epoch, epoch_loss_avg.result(), epoch_accuracy.result()
        ))


# 绘制损失曲线和正确率曲线
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle("Train Metrics")

axes[0].set_ylabel("loss")
axes[0].plot(train_loss_results)

axes[1].set_ylabel("accuracy")
axes[1].set_xlabel("epochs")
axes[1].plot(train_accuracy_results)
plt.show()

# 评估我们的模型
test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
    logits = model(x)
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(predictions, y)

print("Test Accuracy: {:.3%}".format(test_accuracy.result()))


# 使用我们的模型进行预测
prediction_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]  
])

predictions = model(prediction_dataset)

for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    # 概率值
    p = tf.nn.softmax(logits)[class_idx]
    # 名称
    name = class_names[class_idx]
    print("Example {} prediction: {} ({:.1%})".format(
        i, name, p
    ))

