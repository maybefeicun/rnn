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