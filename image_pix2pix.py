# -*- coding: utf8 -*-
# builtin model
import os
import random
import ssl

# third part model
from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

from tensorflow import keras
import tensorflow as tf

# set ssl verify
ssl._create_default_https_context = ssl._create_unverified_context

# enable eager execution
tf.enable_eager_execution()

# local model

print("tensorflow version: {}".format(tf.__version__))

# load data
zip_path = keras.utils.get_file(
    fname="facades.tar.gz",
    origin="https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz",
    extract=True,
    cache_subdir=os.path.abspath(".")    
)

image_dir_path = os.path.join(
    os.path.dirname(zip_path), "facades"
)

print(os.listdir(image_dir_path))

train_dataset_dir_path = os.path.join(image_dir_path, "train")
test_dataset_dir_path = os.path.join(image_dir_path, "test")
val_dataset_dir_path = os.path.join(image_dir_path, "val")

train_dataset_file_paths = (os.path.join(train_dataset_dir_path, fname)
                            for fname in os.listdir(train_dataset_dir_path) if fname.endswith(".jpg"))

test_dataset_file_paths = (os.path.join(test_dataset_dir_path, fname)
                           for fname in os.listdir(test_dataset_dir_path) if fname.endswith(".jpg"))

val_dataset_file_paths = (os.path.join(val_dataset_dir_path, fname)
                          for fname in os.listdir(val_dataset_dir_path) if fname.endswith(".jpg"))

BUFFER_SIZE = 400
IMG_WIDTH = 256
IMG_HEIGHT = 256


def read_image_to_array(file_path):
    return np.array(Image.open(file_path))


train_data_ogn = [read_image_to_array(fp) for fp in train_dataset_file_paths]
test_data_ogn = [read_image_to_array(fp) for fp in test_dataset_file_paths]
val_data_ogn = [read_image_to_array(fp) for fp in val_dataset_file_paths]


def normalized_image(image, istrain):
    height, width, channel = image.shape
    # 左半边为input 右半边为target
    half_width = width // 2
    input_img = image[:, half_width:]
    target_img = image[:, :half_width]

    if istrain:
        # 缩放尺寸
        input_img_resized = tf.image.resize(
            input_img, (286, 286)
        )
        target_img_resized = tf.image.resize(
            target_img, (286, 286)
        )
        # 将两张图片压制到一起
        stack_input_and_target_img = tf.stack([input_img_resized, target_img_resized], axis=0)
        # 随机切分图像
        crop_images = tf.random_crop(stack_input_and_target_img, [2, IMG_HEIGHT, IMG_WIDTH, 3])
        # 分解切分结果
        input_img, target_img = crop_images[0], crop_images[1]
        # 随机翻转图像(水平方向)
        if random.random() > .5:
            input_img = input_img[:, ::-1, :]
            target_img = target_img[:, ::-1, :]
    else:
        # 只进行规定的尺寸缩放和坐标轴扩展
        input_img = tf.image.resize(input_img, (IMG_WIDTH, IMG_HEIGHT)).numpy()
        target_img = tf.image.resize(target_img, (IMG_WIDTH, IMG_HEIGHT)).numpy()

    # 归一化处理
    input_img = input_img / 127.5 - 1
    target_img = target_img / 127.5 - 1
    return input_img, target_img


def img_generate(imgs, istrain=False):
    def gen():
        yield normalized_image(random.choice(imgs), istrain)
    return gen


train_generate = img_generate(train_data_ogn, True)
test_generate = img_generate(test_data_ogn)
val_generate = img_generate(val_data_ogn)


# 格式化所有的图像
train_dataset = tf.data.Dataset.from_generator(
    train_generate, (tf.float32, tf.float32)
).batch(1)

test_dataset = tf.data.Dataset.from_generator(
    test_generate, (tf.float32, tf.float32)
).batch(1)

val_dataset = tf.data.Dataset.from_generator(
    val_generate, (tf.float32, tf.float32)
).batch(1)


# 构建生成器
class DownSample(keras.Model):
    """
    下采样
    """
    def __init__(self, filters, size, apply_batchnormal=True):
        super(DownSample, self).__init__()
        self.apply_batchnormal = apply_batchnormal

        self.conv = keras.layers.Conv2D(
            filters,
            (size, size),
            strides=2,
            padding="same",
            use_bias=False
        )

        if self.apply_batchnormal:
            self.batch_norm = keras.layers.BatchNormalization()

    def call(self, x, training):
        # 卷积
        x = self.conv(x)
        if self.apply_batchnormal:
            x = self.batch_norm(x, training=training)

        # 激活
        x = tf.nn.leaky_relu(x)
        return x


class UpSample(keras.Model):
    """
    上采样
    """
    def __init__(self, filters, size, apply_dropout=False):
        super(UpSample, self).__init__()
        self.apply_dropout = apply_dropout

        self.up_conv = keras.layers.Conv2DTranspose(
            filters,
            (size, size),
            strides=2,
            padding="same",
            use_bias=False
        )

        self.batch_norm = keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = keras.layers.Dropout(.5)

    def call(self, x1, x2, training):
        # 上卷积
        x = self.up_conv(x1)
        x = self.batch_norm(x, training=training)
        # dropout
        if self.apply_dropout:
            x = self.dropout(x)
        # 激活
        x = tf.nn.relu(x)
        x = tf.concat([x, x2], axis=-1)

        return x


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = DownSample(64, 4, apply_batchnormal=False)
        self.down2 = DownSample(128, 4)
        self.down3 = DownSample(256, 4)
        self.down4 = DownSample(512, 4)
        self.down5 = DownSample(512, 4)
        self.down6 = DownSample(512, 4)
        self.down7 = DownSample(512, 4)
        self.down8 = DownSample(512, 4)

        self.up1 = UpSample(512, 4, apply_dropout=True)
        self.up2 = UpSample(512, 4, apply_dropout=True)
        self.up3 = UpSample(512, 4, apply_dropout=True)
        self.up4 = UpSample(512, 4)
        self.up5 = UpSample(256, 4)
        self.up6 = UpSample(128, 4)
        self.up7 = UpSample(64, 4)

        self.last_layer = keras.layers.Conv2DTranspose(
            3,
            (4, 4),
            strides=2,
            padding="same"
        )

    def call(self, x, training):
        # 降采样
        x1 = self.down1(x, training)
        x2 = self.down2(x1, training)
        x3 = self.down3(x2, training)
        x4 = self.down4(x3, training)
        x5 = self.down5(x4, training)
        x6 = self.down6(x5, training)
        x7 = self.down7(x6, training)
        x8 = self.down8(x7, training)

        # 升采样
        x9 = self.up1(x8, x7, training)
        x10 = self.up2(x9, x6, training)
        x11 = self.up3(x10, x5, training)
        x12 = self.up4(x11, x4, training)
        x13 = self.up5(x12, x3, training)
        x14 = self.up6(x13, x2, training)
        x15 = self.up7(x14, x1, training)

        x16 = self.last_layer(x15)
        # 激活
        x16 = tf.nn.tanh(x16)

        return x16


# 构建鉴别器
class DiscDownsample(keras.Model):
    def __init__(self, filters, size, apply_batchnorm=True):
        super(DiscDownsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        self.conv = keras.layers.Conv2D(
            filters,
            (size, size),
            strides=2,
            padding="same",
            use_bias=False
        )
        if self.apply_batchnorm:
            self.batch_norm = keras.layers.BatchNormalization()

    def call(self, x, training):
        # 卷积
        x = self.conv(x)
        if self.apply_batchnorm:
            x = self.batch_norm(x, training=training)

        # 激活
        x = tf.nn.leaky_relu(x)
        return x


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = DiscDownsample(64, 4, False)
        self.down2 = DiscDownsample(128, 4)
        self.down3 = DiscDownsample(256, 4)

        self.zero_pad1 = keras.layers.ZeroPadding2D()

        self.conv = keras.layers.Conv2D(
            512, (4, 4), strides=2, use_bias=False
        )
        self.batch_norm = keras.layers.BatchNormalization()
        self.zero_pad2 = keras.layers.ZeroPadding2D()
        self.last = keras.layers.Conv2D(
            1, (4, 4), strides=1
        )

    def call(self, inp, tar, training):
        x = tf.concat([inp, tar], axis=-1)
        x = self.down1(x, training)
        x = self.down2(x, training)
        x = self.down3(x, training)
        x = self.zero_pad1(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = tf.nn.leaky_relu(x)
        x = self.zero_pad2(x)
        x = self.last(x)

        return x


# 定义损失函数
LAMBDA = 100


def generator_loss_function(gen_out, preb, tar):
    # 采用sigmoid 交叉熵衡量
    # 这里是为了是生成器生成的更像真的而预设的错误，反向训练生成器中的权重
    gen_out_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(gen_out),
        logits=gen_out
    )
    # 计算生成图像的误差 这个误差也是为了使图像更接近原图
    # 但是gen_out_loss 是使权重在顾忌生成器的同时，也估计评估器
    # 让其的权重也能适应评估器，是两者的对抗更有效
    ll_loss = tf.reduce_mean(tf.abs(preb - tar))
    total_loss = gen_out_loss + LAMBDA * ll_loss
    return total_loss


def discriminator_loss_function(preb, tar):
    # 采用 sigmoid 交叉熵衡量 两者之间的相似度
    rel_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(tar),
        logits=tar
    )

    generate_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=np.zeros_like(preb),
        logits=preb
    )

    # 计算总误差
    total_loss = rel_loss + generate_loss
    return total_loss


# 定义优化器
# 生成器优化器
gen_optimizer = tf.train.AdamOptimizer(2e-4, 0.5)
# 评估器优化器
disc_optimizer = tf.train.AdamOptimizer(2e-4, 0.5)


# 开始训练
epoch_num = 10

# 定义生成器和评估器
generator = Generator()
discriminator = Discriminator()


# 查看模型效果
for test_img, test_target in test_dataset.take(1):
    test_img_show = (test_img + 1) * 127.5
    test_target_show = (test_target + 1) * 127.5
    predict_img = (generator(test_img, False) + 1) * 127.5
    fig, axes = plt.subplots(ncols=3, sharey=True)
    axes[0].set_title("test_input")
    axes[0].imshow(test_img_show[0].numpy().astype(np.uint8))
    axes[1].set_title("preb_output")
    axes[1].imshow(predict_img[0].numpy().astype(np.uint8))
    axes[2].set_title("target_output")
    axes[2].imshow(test_target_show[0].numpy().astype(np.uint8))

    plt.show()


# 定义训练流程
def train(dataset, epoches):
    for epoch in range(epoches):
        i = 0
        for input_image, target in dataset.take(100):
            i += 1
            with tf.GradientTape() as gen_gradient, tf.GradientTape() as disc_gradient:
                # 使用生成器计算
                gen_out = generator(input_image, True)

                # 使用评估器计算
                disc_rel_out = discriminator(input_image, target, True)
                disc_gen_out = discriminator(input_image, gen_out, True)

                # 计算误差
                gen_loss = generator_loss_function(disc_gen_out, gen_out, target)
                disc_loss = discriminator_loss_function(disc_gen_out, disc_rel_out)

            # 分别计算生成器和评估器的梯度
            gen_gradient_value = gen_gradient.gradient(
                gen_loss, generator.variables
            )

            disc_gradient_value = disc_gradient.gradient(
                disc_loss, discriminator.variables
            )

            # 使用优化器基于计算的梯度调整变量
            gen_optimizer.apply_gradients(
                zip(gen_gradient_value, generator.variables)
            )
            disc_optimizer.apply_gradients(
                zip(disc_gradient_value, discriminator.variables)
            )
            print("Epoch {}/{} Gen Loss {} DiscLoss {}".format(epoch, epoches, tf.reduce_mean(gen_loss), tf.reduce_mean(disc_loss)))


# 训练10轮
train(train_dataset, 100)

# 查看模型效果
for test_img, test_target in test_dataset.take(1):
    test_img_show = (test_img + 1) * 127.5
    test_target_show = (test_target + 1) * 127.5
    predict_img = (generator(test_img, False) + 1) * 127.5
    fig, axes = plt.subplots(ncols=3, sharey=True)
    axes[0].set_title("test_input")
    axes[0].imshow(test_img_show[0].numpy().astype(np.uint8))
    axes[1].set_title("preb_output")
    axes[1].imshow(predict_img[0].numpy().astype(np.uint8))
    axes[2].set_title("target_output")
    axes[2].imshow(test_target_show[0].numpy().astype(np.uint8))

    plt.show()
