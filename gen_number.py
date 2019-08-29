# -*- coding: utf8 -*-
"""
基于GAN生成手写体文字
"""
import tensorflow as tf
from tensorflow import keras

tf.enable_eager_execution()
print(tf.__version__)

# 加载数据集
train_dataset = None
test_dataset = None


# 构架模型
# 降采样
class DownSample(keras.Model):
    def __init__(self, filters):
        self.conv = keras.layers.Conv2D(
            filters, (4, 4), padding="same"
        )
        self.mp = keras.layers.MaxPool2D()
        self.batch_norm = keras.layers.BatchNormalization()
    
    def call(self, x, training=True):
        x = self.conv(x)
        x = self.conv(x)
        x = self.conv(x)
        if training:
            x = self.batch_norm(x)
        x = tf.nn.relu(x)
        x = self.mp(x)
        return x

# 升采样
class UpSample(keras.Model):
    def __init__(self, filters, rate=0.5):
        self.tconv = keras.layers.Conv2DTranspose(
            filters, (4, 4)
        )

        self.dropout = keras.layers.Dropout(rate)
        self.up = keras.layers.UpSampling2D()

    def call(self, x, training=True):
        x = self.tconv(x)
        x = self.tconv(x)
        x = self.tconv(x)

        if training:
            x = self.dropout(x)
        x = self.mp(x)
        x = tf.nn.relu(x)
        return x


# 构建生成器
class Generator(keras.Model):
    def __init__(self, class_num):
        self.input_layer = keras.layers.Dense(400, input=(None, class_num))

        self.up1 = UpSample(32, 0.5)
        self.up2 = UpSample(256, 0.5)
        self.up3 = UpSample(512, 0.5)
    
    def call(self, x):
        return None


# 构建鉴别器
class Discriminator(keras.Model):
    def __init__(self):
        pass

    def call(self, x):
        return None


# 计算损失
LAMBDA = 100
# 生成器损失
def gen_loss(preb_y, real_y, desc_gen):
    """
    使用评估器的误差，加上自身的生成误差，共同构建
    """
    #
    gen = tf.nn.sigmoid_cross_entropy_with_logits(
        tf.ones_like(desc_gen), logits=desc_gen
    )
    loss = tf.reduce_mean(tf.abs(preb_y - real_y))

    total_loss = loss * LAMBDA + gen

# 鉴别器损失
def desc_loss(gen_y, rel_y):
    rel = tf.ones_like(rel_y)
    err = tf.zeros_like(gen_y)
    rel_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        rel, rel_y
    )
    gene_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        err, gen_y
    )

    total_loss = rel_loss + gene_loss
    return total_loss


num_gen = Generator()
num_desc = Discriminator()

# 优化器
gen_optimizer = tf.train.AdamOptimizer(1e-4)
desc_optimizer = tf.train.AdamOptimizer(1e-4)

# 训练函数
def train(data_set, epoches):
    for epoch in range(epoches):
        for num, real_img in data_set:
            with tf.GradientTape() as gen_t, tf.GradientTape() as desc_t:
                gen_img = num_gen(num)
                desc_num = num_desc(gen_img)
                desc_real_num = num_desc(real_img)

                desc_loss_value = desc_loss(desc_num, desc_real_num)
                gen_loss_value = gen_loss(gen_img, real_img, desc_num)
            gen_gradient = gen_t.gradient(gen_loss_value, num_gen.variables)
            desc_gradient = gen_t.gradient(desc_loss_value, num_desc.variables)

        print("epoch {} desc_loss {} gen_loss {}".format(epoch, desc_loss_value, gen_loss_value))
