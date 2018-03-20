# -*- coding: utf-8 -*-
# @Time    : 2018/3/20 10:46
# @Author  : chen
# @File    : rnn模型进行垃圾短预测.py
# @Software: PyCharm

import os
import re
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile
from tensorflow.python.framework import ops
from tensorflow.contrib import learn

ops.reset_default_graph()

sess = tf.Session()

epochs = 20
batch_size = 250
max_sequence_length = 25
rnn_size = 10
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0005
dropout_keep_prob = tf.placeholder(tf.float32)

# 加载数据集
data_dir = 'temp'
data_file = 'text_data.txt'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.isfile(os.path.join(data_dir, data_file)):
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    # Format Data
    text_data = file.decode()
    text_data = text_data.encode('ascii',errors='ignore')
    text_data = text_data.decode().split('\n')

    # Save data to text file
    with open(os.path.join(data_dir, data_file), 'w') as file_conn:
        for text in text_data:
            file_conn.write("{}\n".format(text))
else:
    # Open data from text file
    text_data = []
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        for row in file_conn:
            text_data.append(row)
    text_data = text_data[:-1]

text_data = [x.split('\t') for x in text_data if len(x)>=1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]

# 清洗数据集
def clean_text(text_string):
    # \s : 匹配任何空白字符，包括空格、制表符、换页符等等。等价于 [ \f\n\r\t\v]
    # ^ : 在中括号中表示不匹配后面的字符，反之为句子开头的匹配
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = ' '.join(text_string.split())
    text_string = text_string.lower()
    return text_string

text_data_train = [clean_text(x) for x in text_data_train]

# 生成字典与句子的标记,注意learn最好这样带入不然有点问题
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_sequence_length,
                                                          min_frequency=min_word_frequency)
# vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length, min_frequency=min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))

# 随机shuffle文本数据集
text_processed = np.array(text_processed) # 感觉与上面有点重复
text_data_target = [1 if target == 'spam' else 0 for target in text_data_target]
shuffled_ix = np.random.permutation(np.arange(len(text_data_target))) # 注意这里多了个premutation与arange函数
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

# 生成训练集与测试集
ix_cutoff = round(len(y_shuffled) * 0.8)
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_) # 获取字典的大小
print("vocabulary size : {:d}".format(vocab_size))