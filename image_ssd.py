# -*- coding -*- 
"""
推断网络
网络输入:
    图像
    类别信息
网络输出:
    正确或者错误
"""
import os
import random

import numpy as np

from PIL import Image

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
tf.enable_eager_execution()
# 定义类别数目
cls_num = 10
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img = np.array(
    Image.open(
        os.path.join(BASE_DIR, "assets/pngs/label/柯西.jpg")
        ).resize((512, 512))
        ).astype(np.float32)
img = np.expand_dims(img, 0)
norm_img = img / 255.0


# 准备训练集
def gen_train_dataset():
    for _ in range(10):
        label_zero = np.zeros((1, 10))
        index_label = random.randint(0, 9)
        is_real = False
        if index_label == 0:
            is_real = True
        label_zero[0][index_label] = 1.0
        yield norm_img, label_zero, is_real


# 搭建模型
class DownSampleBlock(keras.Model):
    def __init__(self, filters, name=None):
        super(DownSampleBlock, self).__init__()
        self.conv1 = keras.layers.Conv2D(
            filters, (4, 4), padding="same"
        )
        self.conv2 = keras.layers.Conv2D(
            filters, (4, 4), padding="same"
        )
        self.conv3 = keras.layers.Conv2D(
            filters, (4, 4), padding="same"
        )
        self.batch_norm = keras.layers.BatchNormalization()
        self.mp = keras.layers.MaxPool2D()

    def call(self, x, batch=True):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        if batch:
            conv3 = self.batch_norm(conv3)
        
        ac_conv = tf.nn.relu(conv3)
        mp = self.mp(ac_conv)
        return mp

class ClsInfoBlock(keras.Model):
    def __init__(self, cls_num, name=None):
        super(ClsInfoBlock, self).__init__()
        self.fl = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(500)
        self.dp1 = keras.layers.Dropout(0.5)
        self.dense2 = keras.layers.Dense(200)
        self.dp2 = keras.layers.Dropout(0.5)
        self.out_dense = keras.layers.Dense(cls_num)

    def call(self, x):
        fl = self.fl(x)
        d1 = self.dense1(fl)
        ac_d1 = tf.nn.relu(d1)
        dp_ac_d1 = self.dp1(ac_d1)
        d2 = self.dense2(dp_ac_d1)
        ac_d2 = tf.nn.relu(d2)
        dp_ac_d2 = self.dp2(ac_d2)
        out_ds = self.out_dense(dp_ac_d2)
        ac_out_ds = tf.nn.sigmoid(out_ds)
        return ac_out_ds

class InferModel(keras.Model):
    def __init__(self, cls_num):
        super(InferModel, self).__init__()
        self.dmp1 = DownSampleBlock(32, "dmp1")
        self.dmp2 = DownSampleBlock(64, "dmp2")
        self.dmp3 = DownSampleBlock(128, "dmp3")
        self.dmp4 = DownSampleBlock(256, "dmp4")
        self.dmp5 = DownSampleBlock(512, "dmp5")
        self.dmp6 = DownSampleBlock(1024, "dmp6")
        self.cls_info_block = ClsInfoBlock(cls_num, "cls_info_layer")
        self.output_dense = keras.layers.Dense(cls_num, name="infer_layer")
        self.concat = keras.layers.Concatenate(-1)

    def call(self, x, cls_infer):
        # 卷积
        dmp1 = self.dmp1(x)
        dmp2 = self.dmp2(dmp1)
        dmp3 = self.dmp3(dmp2)
        dmp4 = self.dmp4(dmp3)
        dmp5 = self.dmp5(dmp4)
        dmp6 = self.dmp6(dmp5)
        
        # 获取分类信息
        cls_info = self.cls_info_block(dmp6)
        con_layer = self.concat([cls_info, cls_infer])
        # 综合推断结果
        out_ = self.output_dense(con_layer)

        return out_

# 定义误差函数
def infer_error_function(infer_output, is_real):
    if is_real:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(infer_output),
            logits=infer_output
        )
    else:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(infer_output),
            logits=infer_output
        )
    total_loss = tf.reduce_mean(tf.square(loss))
    return total_loss


# 定义优化器
infer_optimizer = tf.train.AdamOptimizer(0.01)

# 定义推断模型
infer_model = InferModel(cls_num)

# 定义训练过程
def train(epoches):
    for epoch in range(epoches):
        print("start epoch {}".format(epoch))
        train_set_gen = gen_train_dataset()
        i = 0
        for img, infer, is_real in train_set_gen:
            i += 1
            with tf.GradientTape() as t:
                rel_loss = infer_error_function(
                    infer_model(img, infer), is_real
                )
            # 计算损失梯度
            loss_gradient = t.gradient(rel_loss, infer_model.variables)
            # 优化参数
            infer_optimizer.apply_gradients(
                zip(loss_gradient, infer_model.variables)
            )
            print("epoch {} img {} loss value {}".format(epoch, i, rel_loss))

# 开始训练
train(40)

# 测试模型
for i in range(10):
    zero_label = np.zeros((1, 10))
    zero_label[0][i] = 1.0
    model_output = infer_model(norm_img, zero_label)
    output_mean = tf.reduce_mean(tf.sigmoid(model_output))
    print("label {}".format(zero_label))
    print("output {}".format(output_mean))
