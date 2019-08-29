# -*- coding: utf8 -*-
# 机器学习翻译模型
import os
import unicodedata

import tensorflow as tf
from tensorflow import keras


print(tf.__version__)


path_to_zip = keras.utils.get_file(
    fname="spa-eng.zip", origin="http://download.tensorflow.org/data/spa-eng.zip",
    extract=True
)

path_to_file = os.path.join(os.path.dirname(path_to_zip), "spa-eng/spa.txt")

# 从unicode解析为ascii码的码值
def unicode_to_ascii(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) 
        if unicodedata.category(c) != "Mn")


# 预处理句子的函数
def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.rstrip().strip()

    w = "<start> " + w + " <end>"
    return w

# 创建数据集
def create_dataset(path, num_examples):
    lines = open(path, "r", encoding="utf-8").read().strip().split("\n")

    word_pairs = [[preprocess_sentence(w) for w in l.split("\t")]for l in lines[:num_examples]]
    
    return word_pairs

# 创建语言的index类
class LanguageIndex():
    def __init__(self):
        pass

    def create_index(self):
        pass

# 计算tensor的长度
def max_length(tensor):
    return max(len(t) for t in tensor)

# 设置加载数据集
def load_dataset(path, num_examples):
    pass

# 加载数据集
num_examples = 30000

# ___ = load_dataset(path_to_file, num_examples)

# 数据集分包

# 生成标准的tensorflow 数据集对象

# 定义编码器
class Encoder(tf.keras.Model):
    def __init__(self):
        pass
    
    def call(self):
        pass

    def initialize_hidden_state(self)
        pass

# 定义解码器
class Decoder():
    def __init__(self):
        pass

    def call(self):
        pass

    def initialize_hidden_state():
        pass


# 定义损失函数
def loss_fuction(preb_y, real_y):
    pass


# 定义训练相关的参数

# 评估模型

