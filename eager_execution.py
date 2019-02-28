# -*- coding: utf8 -*-
import time

import tensorflow as tf

import tempfile

import numpy as np

tf.enable_eager_execution()

# use tensor and some op
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(9))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello~"))

print(tf.square(9) + tf.square(8))
# some tensor attribute
a_tensor = tf.matmul([[1]], [[2, 3]])
print("tensor shape: {}".format(a_tensor.shape))
print("tensor type: {}".format(a_tensor.dtype))

# convert np.array and tensor
ndarry = np.ones((3, 3))

print("Use Tensor Method Operate Ndarry")
tensor = tf.multiply(ndarry, 42)
print("Tensor: \n{}\ntype: {}".format(tensor, tensor.dtype))

print("Use Ndarry Method Operate Tensor")
ndarry_add = np.add(tensor, 1)
print("ndarry_add: \n{}\ntype: {}".format(ndarry_add, ndarry_add.dtype))

print("Get Numpy Object From Tensor")
print("narray: \n{}\ntype: {}".format(tensor.numpy(), tensor.numpy().dtype))

# 查看gpu是否可用
x = tf.random_uniform((3, 3))
print("Gpu Available: {}".format(tf.test.is_gpu_available()))
print("Tensor Use {}".format(x.device))

# 控制Tensor的放置位置
def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)
    result = time.time() - start
    print("10 Loops: {:0.2f}ms".format(result * 1000))

# 测试在cpu上的效率
print("On Cpu")
with tf.device("CPU:0"):
    x = tf.random_uniform((1000, 1000))
    assert x.device.endswith("CPU:0")
    time_matmul(x)

# 测试在gpu上的效率
if tf.test.is_gpu_available():
    print("On Gpu")
    with tf.device("GPU:0"):
        x = tf.random_uniform((1000, 1000))
        assert x.device.endswith("GPU:0")
        time_matmul(x)

# 数据集的管理
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
print("ds_tensors: {}".format(ds_tensors))

# Create CSV file
_, filename = tempfile.mkstemp()
with open(filename, "w") as f:
    f.write("Line1\nLine2\nLine3")

ds_file = tf.data.TextLineDataset(filename)
print("ds_file: {}".format(ds_file))

# 处理前输出
print("ds_tensors(before operate): ~")
for tensor in ds_tensors:
    print(tensor)

print("ds_file(before operate): ~")
for tensor in ds_file:
    print(tensor)

# 将一些操作应用到数据集中
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

print("ds_tensors: ~")
for tensor in ds_tensors:
    print(tensor)

print("ds_file: ~")
for tensor in ds_file:
    print(tensor)
