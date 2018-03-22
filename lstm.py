# -*- coding: utf-8 -*-
# @Time    : 2018/3/21 13:52
# @Author  : chen
# @File    : lstm.py
# @Software: PyCharm

import os
import re
import string
import requests
import numpy as np
import collections
import random
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

sess = tf.Session()

batch_size = 100
learning_rate = 0.001
min_word_freq = 5
rnn_size = 128
epochs = 10
training_seq_len = 50
embedding_size = rnn_size
save_every = 500
eval_every = 50
prime_texts = ['thou art more', 'to be or not to', 'wherefore art thou']

data_dir = 'temp'
data_file = 'shakespeare.txt'
model_path = 'shakespeare_model'
full_model_dir = os.path.join(data_dir, model_path)
# 我们要保留连字符与省略符
punctuation = string.punctuation
punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])

# 下载数据集
if not os.path.exists(full_model_dir):
    os.makedirs(full_model_dir)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if os.path.isfile(os.path.join(data_dir, data_file)):
    # 文件存在
    with open(os.path.join(data_dir, data_file), 'r') as file_read:
        s_text = file_read.read().replace('\n', '') # 这个写法值得学习

s_text = re.sub(r'[{}]'.format(punctuation), '', s_text) # 这个写法厉害了
s_text = re.sub('\s+', ' ', s_text).strip().lower() # 注意\s的写法

def build_vocab(text, min_word_freq):
    word_counts = collections.Counter(text.split(' '))
    # 卧槽这个写法有点厉害了
    # for key, val in word_counts不能直接遍历
    # for key, val in word_counts.items():
    #     if val > min_word_freq:
    #         word_counts[key] = val
    word_counts = {key: val for key, val in word_counts.items() if val > min_word_freq}
    words = word_counts.keys()
    vocab_to_ix_dict = {key: (ix + 1) for ix, key in enumerate(words)} # ix+1是为了留一个给unknow
    vocab_to_ix_dict['unkown'] = 0
    ix_to_vocab_dict = {val: key for key, val in vocab_to_ix_dict.items()} # 在球员给反向字典

    return ix_to_vocab_dict, vocab_to_ix_dict

ix2vocab, vocab2ix = build_vocab(s_text, min_word_freq)
vocab_size = len(ix2vocab) + 1 # 这里没看懂为什么要加一

# 将文本内容转换为索引数组，这个是给整个文件进行索引标记
s_text_words = s_text.split(' ')
s_text_ix = []
for ix, x in enumerate(s_text_words):
    try: # 这里在处理没出现的单词的时候采用了try/except的方法
        s_text_ix.append(vocab2ix[x])
    except:
        s_text_ix.append(0)

class LSTM_Model():
    def __init__(self, rnn_size, batch_size, learning_rate,
                 training_seq_len, vocab_size, infer=False):
        self.rnn_size = rnn_size
        # self.batch_size = batch_size
        self.learning_rate = learning_rate
        # self.training_seq_len = training_seq_len
        self.vocab_size = vocab_size
        self.infer = infer

        if infer:
            self.batch_size = 1
            self.training_seq_len = 1
        else:
            self.batch_size = batch_size
            self.training_seq_len = training_seq_len

        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

        self.x_data = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        self.y_output = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])

        with tf.variable_scope('lstm_vars'):
            # softmax Output Weights
            W = tf.get_variable('W', [self.rnn_size, self.vocab_size], tf.float32, tf.random_normal_initializer())
            b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.random_normal_initializer())

            # 定义词嵌入矩阵
            embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.rnn_size], tf.float32, tf.random_normal_initializer())
            embedding_output = tf.nn.embedding_lookup(embedding_mat, self.x_data)
            '''
            split(value, num_or_size_splits, axis=0, num=None, name='split')
            该函数的作用就是将输入张量value按照num_or_size_splits与axis的样式进行切分
            其中，num_or_size_splits为划分的形式有可能是整数则是均分，也肯能是list形式则按list中的数据进行分
            axis则是规定划分的维度
            
            例如：
            split(tensor_a, 3, 1),其中tensor_a的形状是[3, 6, 1]
            则最后划分的结果为三个张量均为shape=[3, 2, 1]的形式
            
            注意：num_or_size_splits一定要被value.shape()[axis]整除
            '''
            rnn_inputs = tf.split(1, self.training_seq_len, embedding_output)
            rnn_inputs_trimmed = [tf.squeeze(x, [1]) for x in rnn_inputs]

            def inferred_loop(prev, count):
                prev_transformed = tf.matmul(prev, W) + b
                prev_symbol = tf.stop_gradient(tf.argmax(prev_transformed, 1))
                output = tf.nn.embedding_lookup(embedding_mat, prev_symbol)

                return output

            decoder = tf.nn.seq2seq