# -*- coding: utf8 -*-
import tensorflow as tf


print(tf.__version__)
tf.enable_eager_execution()

# 关于层的一些探究
layer = tf.keras.layers.Dense(units=10, input_shape=(None, 5))

layer(tf.zeros((10, 5)))
# 这里边包含了初始化的权重和偏置
print("Veriable: \n{}".format(layer.variables))

print("Weights: \n{}".format(layer.kernel))
print("Bias: \n{}".format(layer.bias))

# 自定义一个层
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_output):
        super(MyDenseLayer, self).__init__()
        self.num_output = num_output
    
    def build(self, input_shape):
        self.kernel = self.add_variable(
            "kernel", shape=[
                int(input_shape[-1]),
                self.num_output
            ]
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

layer = MyDenseLayer(10)

print(layer(tf.zeros((10, 5))))
print(layer.trainable_variables)

# 组建自己的层组
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filter1, filter2, filter3 = filters

        self.conv1 = tf.keras.layers.Conv2D(filter1, (1, 1))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filter2, kernel_size, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filter3, (1, 1))
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)

block = ResnetIdentityBlock(1, [1, 1, 1])
print("block variables: {}".format(block(tf.zeros((1, 2, 3, 3)))))
print("block name: {}".format([x.name for x in block.trainable_variables]))

# 定义一个常用的序贯模型
seq = tf.keras.Sequential(
    layers=[
        tf.keras.layers.Conv2D(1, (1, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            2, 1, padding="same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(3, (1, 1)),
        tf.keras.layers.BatchNormalization()
    ]
)

seq(tf.zeros((1, 2, 3, 3)))
print("seq: {}".format(seq))
print("seq_variables: {}".format(seq.trainable_variables))
