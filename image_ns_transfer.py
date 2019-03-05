# -*- coding: utf8 -*-
import os

from PIL import Image

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

import numpy as np

print(tf.__version__)
tf.enable_eager_execution()

# 加载图像
def load_img(path_to_img):
    return Image.open(path_to_img)

# 显示图像
def show_img(img, title=None):
    axes = plt.subplot(111)
    if title:
        axes.set_title(title)
    axes.imshow(img)
    plt.show()

# 预处理图像
def process_img(img):
    new_img = img.resize((512, 512))
    img_array = np.array(new_img).astype(np.float32) / 255.0
    return img_array

def gen_dir_imgs(dir_path="."):

    def gen():
        dir_abs_path = os.path.abspath(dir_path)
        file_list = os.listdir(dir_abs_path)
        filter_img = filter(
            lambda fname: fname.endswith(".jpg") or
            fname.endswith(".png")
            , file_list)
        
        img_file_path = map(lambda fn: os.path.join(dir_abs_path, fn), filter_img)
        for path in img_file_path:
            yield process_img(load_img(path))
    return gen
    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_gen = gen_dir_imgs(os.path.join(BASE_DIR, "assets/pngs/train"))
test_gen = gen_dir_imgs(os.path.join(BASE_DIR, "assets/pngs/test"))
label_gen = gen_dir_imgs(os.path.join(BASE_DIR, "assets/pngs/label"))

train_data_set = tf.data.Dataset.from_generator(
    train_gen, output_types=(tf.float32)
).batch(1)

test_data_set = tf.data.Dataset.from_generator(
    test_gen, output_types=(tf.float32)
).batch(1)

label_data_set = tf.data.Dataset.from_generator(
    label_gen, output_types=(tf.float32)
).batch(1)

# 设计模型
class NSModel(keras.Model):
    def __init__(self):
        super(NSModel, self).__init__()

        self.input_ = keras.layers.Conv2D(
            32, (4, 4), padding="same")
    
        self.conv = keras.layers.Conv2D(32, (4, 4), padding="same")
        self.dconv = keras.layers.Conv2DTranspose(32, (4, 4), padding="same")

        self.output_ = keras.layers.Conv2D(3, (4, 4),padding="same")

    def call(self, x1, x2=None):
        x1_ = self.input_(x1)
        x1_ = self.conv(x1_)
        x1_ = self.conv(x1_)
        x1_ = self.conv(x1_)
        x1_ = self.conv(x1_)
        x1_ = self.conv(x1_)
        x1_ = self.conv(x1_)

        if x2 is not None:
            x2_ = self.input_(x2)
            x2_ = self.conv(x2_)
            x2_ = self.conv(x2_)
            x2_ = self.conv(x2_)
            x2_ = self.conv(x2_)
            x2_ = self.conv(x2_)
            x2_ = self.conv(x2_)
            x1_ = (x1_ + x2_) / 2
        x1_ = self.dconv(x1_)
        x1_ = self.dconv(x1_)
        x1_ = self.dconv(x1_)
        x1_ = self.dconv(x1_)
        x1_ = self.dconv(x1_)
        x1_ = self.dconv(x1_)
        x1_ = self.dconv(x1_)

        output_x = self.output_(x1_)
        activate_x = tf.nn.tanh(output_x)
        return activate_x
        

# 定义损失函数
def loss_function(preb_y, real_y):
    return tf.reduce_mean(tf.square(real_y - preb_y))

# 定义模型
model = NSModel()

# 定义优化器
optimizer = tf.train.AdamOptimizer(0.1)

# 定义训练
def train(train_set, epoches):
    for train_img in train_data_set:
        for epoch in range(epoches):
            with tf.GradientTape() as t:
                loss_value = loss_function(model(train_img) ,train_img)
            optimizer.apply_gradients(
                zip(
                    t.gradient(loss_value, model.variables), model.variables
                )
            )
            print("epoch {} loss {}".format(epoch, loss_value))

train(train_data_set, 10)

# 评估模型
# ---
for t1 in test_data_set.take(1):
    for t2 in label_data_set.take(1):
        output = model(t1, t2)

# 查看输出
# ---
print(output)
plt.imshow((255 * output).numpy().astype(np.uint8)[0])
plt.show()
