# -*- coding: utf-8 -*-
# @Time    : 2018/3/21 13:52
# @Author  : chen
# @File    : lstm.py
# @Software: PyCharm

"""
从这段代码中，我明白了有些东西不需要自己能看懂，只需要能够使用即可
比如这段代码中的class LSTM这个类，到时候就直接拿来用就行
"""

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
from tensorflow.contrib import legacy_seq2seq

sess = tf.Session()

batch_size = 100
learning_rate = 0.001
min_word_freq = 5
rnn_size = 128 # rnn的节点数
epochs = 5
training_seq_len = 50 # 一次训练50行？
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

# 建立字典
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
# 这个初始化字典与反向字典
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


# Define LSTM RNN Model
# 代码的核心部分，构建了一个lstm的类，我们通过这个类进行多次训练批量数据和臭氧生成的文本
class LSTM_Model():
    def __init__(self, embedding_size, rnn_size, batch_size, learning_rate,
                 training_seq_len, vocab_size, infer_sample=False):
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.vocab_size = vocab_size
        self.infer_sample = infer_sample
        self.learning_rate = learning_rate

        if infer_sample:
            self.batch_size = 1
            self.training_seq_len = 1
        else:
            self.batch_size = batch_size
            self.training_seq_len = training_seq_len

        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32) # 设置初始状态

        self.x_data = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        self.y_output = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        '''
        先通过tf.variable_scope生成一个上下文管理器，
        并指明需求的变量在这个上下文管理器中，就可以直接通过tf.get_variable获取已经生成的变量。
        tf.get_variable函数，变量名称是一个必填的参数，它会根据变量名称去创建或者获取变量。
        '''
        with tf.variable_scope('lstm_vars'): # 注意下这里面的变量在定义时就已经被初始化了
            # Softmax Output Weights
            W = tf.get_variable('W', [self.rnn_size, self.vocab_size], tf.float32, tf.random_normal_initializer())
            b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.constant_initializer(0.0))

            # Define Embedding
            embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.embedding_size],
                                            tf.float32, tf.random_normal_initializer())
            # 这是一个常规写法
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


            tf.squeeze(input, axis)
            # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
            tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]
            '''
            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.training_seq_len, value=embedding_output) # run_inputs为一个列表
            rnn_inputs_trimmed = [tf.squeeze(x, [1]) for x in rnn_inputs] # run_inputs_trimmed被压缩为一个[100, 128]的矩阵

        # If we are inferring (generating text), we add a 'loop' function
        # Define how to get the i+1 th input from the i th output
        def inferred_loop(prev, count):
            # Apply hidden layer
            prev_transformed = tf.matmul(prev, W) + b
            # Get the index of the output (also don't run the gradient)
            prev_symbol = tf.stop_gradient(tf.argmax(prev_transformed, 1))
            # Get embedded vector
            output = tf.nn.embedding_lookup(embedding_mat, prev_symbol)
            return (output)

        # decoder = tf.contrib.legacy_seq2seq.rnn_decoder
        '''
        本段代码使用了tf.contrib.legacy_seq2seq.rnn_decoder方法
        第一个参数decoder_inputs=rnn_inputs_trimmed,输入列表长度为num_steps,每个元素是[batch_size, input_size]的2-D维的tensor
        第二个参数initial_state=self.initial_state,初始化状态，2-D的tensor，shape为 [batch_size x cell.state_size]
        第三个参数cell
        第四个参数loop_function如果不为空，则将该函数应用于第i个输出以得到第i+1个输入，此时decoder_inputs变量除了第一个元素之外其他元素会被忽略。
            其形式定义为：loop(prev, i)=next。prev是[batch_size x output_size]，i是表明第i步，next是[batch_size x input_size]
        '''
        decoder = legacy_seq2seq.rnn_decoder
        outputs, last_state = decoder(rnn_inputs_trimmed,
                                      self.initial_state,
                                      self.lstm_cell,
                                      loop_function=inferred_loop if infer_sample else None)
        # Non inferred outputs
        '''
        t1 = [[1, 2, 3], [4, 5, 6]]
        t2 = [[7, 8, 9], [10, 11, 12]]
        tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
        '''
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.rnn_size])
        # Logits and output
        self.logit_output = tf.matmul(output, W) + b
        self.model_output = tf.nn.softmax(self.logit_output)

        loss_fun = legacy_seq2seq.sequence_loss_by_example
        loss = loss_fun([self.logit_output], [tf.reshape(self.y_output, [-1])],
                        [tf.ones([self.batch_size * self.training_seq_len])],
                        self.vocab_size)
        self.cost = tf.reduce_sum(loss) / (self.batch_size * self.training_seq_len)
        self.final_state = last_state
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))

    def sample(self, sess, words=ix2vocab, vocab=vocab2ix, num=10, prime_text='thou art'):
        state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
        word_list = prime_text.split()
        for word in word_list[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed_dict=feed_dict)

        out_sentence = prime_text
        word = word_list[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed_dict = {self.x_data: x, self.initial_state: state}
            [model_output, state] = sess.run([self.model_output, self.final_state], feed_dict=feed_dict)
            sample = np.argmax(model_output[0])
            if sample == 0:
                break
            word = words[sample]
            out_sentence = out_sentence + ' ' + word
        return (out_sentence)


# Define LSTM Model
lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate,
                        training_seq_len, vocab_size)

# Tell TensorFlow we are reusing the scope for the testing
# 注意下当前的variable_scope
with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    test_lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate,
                                 training_seq_len, vocab_size, infer_sample=True)

# Create model saver
# saver的用法需要十分注意#####
saver = tf.train.Saver(tf.global_variables())

# Create batches for each epoch
num_batches = int(len(s_text_ix) / (batch_size * training_seq_len)) + 1
# Split up text indices into subarrays, of equal size
batches = np.array_split(s_text_ix, num_batches)
# Reshape each split into [batch_size, training_seq_len]
batches = [np.resize(x, [batch_size, training_seq_len]) for x in batches]

# Initialize all variables
init = tf.global_variables_initializer()
sess.run(init)

# Train model
train_loss = []
iteration_count = 1
for epoch in range(epochs):
    # Shuffle word indices
    random.shuffle(batches)
    # Create targets from shuffled batches
    targets = [np.roll(x, -1, axis=1) for x in batches]
    # Run a through one epoch
    print('Starting Epoch #{} of {}.'.format(epoch + 1, epochs))
    # Reset initial LSTM state every epoch
    state = sess.run(lstm_model.initial_state)
    write = tf.summary.FileWriter('./lstm_graph', sess.graph)
    for ix, batch in enumerate(batches):
        training_dict = {lstm_model.x_data: batch, lstm_model.y_output: targets[ix]}
        c, h = lstm_model.initial_state
        training_dict[c] = state.c
        training_dict[h] = state.h

        temp_loss, state, _ = sess.run([lstm_model.cost, lstm_model.final_state, lstm_model.train_op],
                                       feed_dict=training_dict)
        train_loss.append(temp_loss)

        # Print status every 10 gens
        if iteration_count % 10 == 0:
            summary_nums = (iteration_count, epoch + 1, ix + 1, num_batches + 1, temp_loss)
            print('Iteration: {}, Epoch: {}, Batch: {} out of {}, Loss: {:.2f}'.format(*summary_nums))

        # Save the model and the vocab
        if iteration_count % save_every == 0:
            # Save model
            model_file_name = os.path.join(full_model_dir, 'model')
            saver.save(sess, model_file_name, global_step=iteration_count)
            print('Model Saved To: {}'.format(model_file_name))
            # Save vocabulary
            dictionary_file = os.path.join(full_model_dir, 'vocab.pkl')
            with open(dictionary_file, 'wb') as dict_file_conn:
                pickle.dump([vocab2ix, ix2vocab], dict_file_conn)

        if iteration_count % eval_every == 0:
            for sample in prime_texts:
                print(test_lstm_model.sample(sess, ix2vocab, vocab2ix, num=10, prime_text=sample))

        iteration_count += 1
    write.close()

# Plot loss over time
plt.plot(train_loss, 'k-')
plt.title('Sequence to Sequence Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

# class LSTM_Model():
#     def __init__(self, rnn_size, batch_size, learning_rate,
#                  training_seq_len, vocab_size, infer_sample=False):
#         self.rnn_size = rnn_size
#         # self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         # self.training_seq_len = training_seq_len
#         self.vocab_size = vocab_size
#         self.infer_sample = infer_sample
#
#         if infer_sample:
#             self.batch_size = 1
#             self.training_seq_len = 1
#         else:
#             self.batch_size = batch_size
#             self.training_seq_len = training_seq_len
#
#         self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
#         self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)
#
#         self.x_data = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
#         self.y_output = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
#
#         with tf.variable_scope('lstm_vars'):
#             # softmax Output Weights
#             W = tf.get_variable('W', [self.rnn_size, self.vocab_size], tf.float32, tf.random_normal_initializer())
#             b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.random_normal_initializer())
#
#             # 定义词嵌入矩阵
#             embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.rnn_size], tf.float32, tf.random_normal_initializer())
#             embedding_output = tf.nn.embedding_lookup(embedding_mat, self.x_data)
#             '''
#             split(value, num_or_size_splits, axis=0, num=None, name='split')
#             该函数的作用就是将输入张量value按照num_or_size_splits与axis的样式进行切分
#             其中，num_or_size_splits为划分的形式有可能是整数则是均分，也肯能是list形式则按list中的数据进行分
#             axis则是规定划分的维度
#
#             例如：
#             split(tensor_a, 3, 1),其中tensor_a的形状是[3, 6, 1]
#             则最后划分的结果为三个张量均为shape=[3, 2, 1]的形式
#
#             注意：num_or_size_splits一定要被value.shape()[axis]整除
#
#
#             tf.squeeze(input, axis)
#             # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
#             tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]
#             '''
#             rnn_inputs = tf.split(embedding_output, self.training_seq_len, 1)
#             rnn_inputs_trimmed = [tf.squeeze(x, [1]) for x in rnn_inputs]
#
#         def inferred_loop(prev, count):
#             prev_transformed = tf.matmul(prev, W) + b
#             prev_symbol = tf.stop_gradient(tf.argmax(prev_transformed, 1))
#             output = tf.nn.embedding_lookup(embedding_mat, prev_symbol)
#
#             return output
#         # from tensorflow.contrib import legacy_seq2seq
#         decoder = legacy_seq2seq.rnn_decoder
#         outputs, last_state = decoder(rnn_inputs_trimmed,
#                                       self.initial_state,
#                                       self.lstm_cell,
#                                       loop_function=inferred_loop if infer_sample else None)
#
#         output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.rnn_size])
#         # Logits and output
#         self.logit_output = tf.matmul(output, W) + b
#         self.model_output = tf.nn.softmax(self.logit_output)
#
#         loss_fun = tf.contrib.legacy_seq2seq.sequence_loss_by_example
#         loss = loss_fun([self.logit_output], [tf.reshape(self.y_output, [-1])],
#                         [tf.ones([self.batch_size * self.training_seq_len])],
#                         self.vocab_size)
#         self.cost = tf.reduce_sum(loss) / (self.batch_size * self.training_seq_len)
#         self.final_state = last_state
#         gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 4.5)
#         optimizer = tf.train.AdamOptimizer(self.learning_rate)
#         self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
#
#     def sample(self, sess, words=ix2vocab, vocab=vocab2ix, num=10, prime_text='thou art'):
#         state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
#         word_list = prime_text.split()
#         for word in word_list[:-1]:
#             x = np.zeros((1, 1))
#             x[0, 0] = vocab[word]
#             feed_dict = {self.x_data: x, self.initial_state: state}
#             [state] = sess.run([self.final_state], feed_dict=feed_dict)
#
#         out_sentence = prime_text
#         word = word_list[-1]
#         for n in range(num):
#             x = np.zeros((1, 1))
#             x[0, 0] = vocab[word]
#             feed_dict = {self.x_data: x, self.initial_state: state}
#             [model_output, state] = sess.run([self.model_output, self.final_state], feed_dict=feed_dict)
#             sample = np.argmax(model_output[0])
#             if sample == 0:
#                 break
#             word = words[sample]
#             out_sentence = out_sentence + ' ' + word
#         return (out_sentence)
#
# # Define LSTM Model
# lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate,
#                         training_seq_len, vocab_size)
#
# # Tell TensorFlow we are reusing the scope for the testing
# with tf.variable_scope(tf.get_variable_scope(), reuse=True):
#     test_lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate,
#                                  training_seq_len, vocab_size, infer_sample=True)
#
# # Create model saver
# saver = tf.train.Saver(tf.global_variables())
#
# # Create batches for each epoch
# num_batches = int(len(s_text_ix) / (batch_size * training_seq_len)) + 1
# # Split up text indices into subarrays, of equal size
# batches = np.array_split(s_text_ix, num_batches)
# # Reshape each split into [batch_size, training_seq_len]
# batches = [np.resize(x, [batch_size, training_seq_len]) for x in batches]
#
# # Initialize all variables
# init = tf.global_variables_initializer()
# sess.run(init)
#
# # Train model
# train_loss = []
# iteration_count = 1
# for epoch in range(epochs):
#     # Shuffle word indices
#     random.shuffle(batches)
#     # Create targets from shuffled batches
#     targets = [np.roll(x, -1, axis=1) for x in batches]
#     # Run a through one epoch
#     print('Starting Epoch #{} of {}.'.format(epoch + 1, epochs))
#     # Reset initial LSTM state every epoch
#     state = sess.run(lstm_model.initial_state)
#     for ix, batch in enumerate(batches):
#         training_dict = {lstm_model.x_data: batch, lstm_model.y_output: targets[ix]}
#         c, h = lstm_model.initial_state
#         training_dict[c] = state.c
#         training_dict[h] = state.h
#
#         temp_loss, state, _ = sess.run([lstm_model.cost, lstm_model.final_state, lstm_model.train_op],
#                                        feed_dict=training_dict)
#         train_loss.append(temp_loss)
#         # Print status every 10 gens
#         if iteration_count % 10 == 0:
#             summary_nums = (iteration_count, epoch + 1, ix + 1, num_batches + 1, temp_loss)
#             print('Iteration: {}, Epoch: {}, Batch: {} out of {}, Loss: {:.2f}'.format(*summary_nums))
#
#         # Save the model and the vocab
#         if iteration_count % save_every == 0:
#             # Save model
#             model_file_name = os.path.join(full_model_dir, 'model')
#             saver.save(sess, model_file_name, global_step=iteration_count)
#             print('Model Saved To: {}'.format(model_file_name))
#             # Save vocabulary
#             dictionary_file = os.path.join(full_model_dir, 'vocab.pkl')
#             with open(dictionary_file, 'wb') as dict_file_conn:
#                 pickle.dump([vocab2ix, ix2vocab], dict_file_conn)
#
#         if iteration_count % eval_every == 0:
#             for sample in prime_texts:
#                 print(test_lstm_model.sample(sess, ix2vocab, vocab2ix, num=10, prime_text=sample))
#
#         iteration_count += 1
#
# # Plot loss over time
# plt.plot(train_loss, 'k-')
# plt.title('Sequence to Sequence Loss')
# plt.xlabel('Generation')
# plt.ylabel('Loss')
# plt.show()