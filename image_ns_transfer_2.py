# -*- coding: utf8 -*-
import tensorflow as tf
from tensorflow import keras

print("tensorflow version: {}".format(tf.__version__))
tf.enable_eager_execution()

# 构建网络

# some block
class DownsampleBlock(keras.Model):
    def init(self, filters, kernel_size=(4, 4)):
        self.conv_1 = keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size
        )

        self.conv_2 = keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size
        )

        self.conv_3 = keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size
        )
        self.batch_norm = keras.layers.BatchNormalization()
        self.mp = keras.layers.MaxPool2D()

    def call(self, x, training=True):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        if training:
            x = self.batch_norm(x)

        x = self.mp(x)
        x = tf.nn.relu(x)
        return x

class UpsampleBlock(keras.Model):
    def init(self, filters, kernel_size=(4, 4)):
        self.dconv1 = keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=kernel_size
        )    

        self.dconv2 = keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=kernel_size
        )

        self.dconv3 = keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=kernel_size
        )

        self.batch_norm = keras.layers.BatchNormalization()
        self.up = keras.layers.UpSampling2D()

    def call(self, x, training=True):
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        if training:
            x = self.batch_norm(x)
        x = self.up(x)
        x = tf.nn.relu(x)
        return x

# 构建一个生成器
class Generator(keras.Model):
    pass

# 构建一个评估器
class Appreciator(keras.Model):
    pass

# model
class GAAModel(keras.Model):
    def init(self):
        self.gen = Generator()
        # 左图评估器
        self.app1 = Appreciator()
        # 右图鉴赏器
        self.app2 = Appreciator()

        # 归并
        self.concat = keras.layers.Concatenate()
    
    def build(self):
        # 返回评估器和鉴赏器
        return self.gen, self.app1, self.app2

    def call(self, left, right, training=True):
        x = self.concat(left, right)
        # 生成图像
        x_gen = self.gen(x)

        # 鉴赏图像
        app_1_score = self.app1(x_gen)
        app_2_score = self.app2(x_gen)

        # 鉴赏左图和右图
        app_1_true = self.app1(left)
        app_1_false = self.app1(right)

        app_2_true = self.app2(right)
        app_2_false = self.app2(left)

        return x_gen, app_1_score, app_2_score, \
            app_1_true, app_2_true, app_1_false, app_2_false

# 构建模型
gaa_model = GAAModel()
gen, app1, app2 = gaa_model.build()

# 计算损失
# app 损失
def app_loss(true_score, false_score, gen_score, pred_score):
    # true loss
    true_loss = tf.reduce_mean(tf.square(true_score - tf.ones_like(true_score)))
    # false loss
    false_loss = tf.reduce_mean(tf.square(false_score - tf.zeros_like(false_score)))
    # real loss
    real_loss = tf.reduce_mean(tf.square(gen_score - pred_score))

    total_loss = real_loss + true_loss + false_loss
    return total_loss

# 计算总损失
def loss(app1_score, app2_score):
    pass
