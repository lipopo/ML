# -*- coding: utf8 -*-
# 自定义训练
import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import tensorflow as tf


print(tf.__version__)
tf.enable_eager_execution()

x = tf.zeros((10, 10))
x += 2
print("Tensor \n{}\n".format(x))

# 演示对于变量的值操作
v = tf.Variable(1.0)
assert v.numpy() == 1.0

v.assign(3.0)
assert v.numpy() == 3.0

v.assign(tf.square(v))
assert v.numpy() == 9.0

# 训练线性模型
# 1.定义一个模型
# 2.定义一个残差函数
# 3.获取训练数据
# 4.运行模型生产数据,计算残差并使用优化器优化

# 定义一个模型
class Model(object):
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(2.0)
    
    def __call__(self, x):
        return self.W * x + self.b

model = Model()
assert model(3.0).numpy() == 17.0

# 定义一个误差函数
def error_function(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

# 生成训练数据
TRUE_W = 3.0
TRUE_b = 5.0

NUM_EXAMPLES = 1000

inputs = tf.random_normal(shape=[NUM_EXAMPLES])
noise = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = TRUE_W * inputs + TRUE_b + noise

train_dataset = pd.DataFrame(data={"inputs": inputs, "outputs": outputs})

sns.lmplot("inputs", "outputs", data=train_dataset)
plt.show()

# 定义训练和参数调整
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = error_function(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

# 开始训练
model = Model()

Ws, bs = [], []
losses = []
learning_rate = 0.1
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = error_function(model(inputs), outputs)
    losses.append(current_loss)
    train(model, inputs, outputs, learning_rate)
    print(
        "Epochs %d W=%1.2f b=%1.2f loss=%2.5f" % (
            epoch, Ws[-1], bs[-1], losses[-1]
        )
    )

train_set = pd.DataFrame(data={
    "epoch": epochs,
    "w": Ws,
    "b": bs,
    "losses": losses
})

ax = plt.subplot(211)
ax.plot(epochs, Ws, color='b', label="Train_W")
ax.plot(epochs, [TRUE_W] * len(epochs), 'b--', label="True_W")
ax.plot(epochs, bs, color='r', label="Train_b")
ax.plot(epochs, [TRUE_b] * len(epochs), 'r--', label="True_b")
plt.legend()

ax2 = plt.subplot(212)
ax2.plot(epochs, losses, 'b', label="Loss")
plt.legend()
plt.show()
