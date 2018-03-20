# -*- coding: utf-8 -*-
# @Time    : 2018/3/20 10:46
# @Author  : chen
# @File    : rnn模型进行垃圾短预测.py
# @Software: PyCharm

'''
本代码主要取理解一下rnn的基本过程
主要的函数：
1.tf.nn.rnn_cell.BasicRNNCell（）：初始化一个rnn_cell
2.tf.nn.dynamic_rnn（）：一次性执行n次的rnn过程，得到所有过程的输出结果与状态结果
3.tf.nn.dropout（）：处理过拟合的情况
4.tf.gather()：用来取张量的某一维的值
'''

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

epochs = 20 # 训练数据为20个时期
batch_size = 250 # 一次训练250个数据
max_sequence_length = 25 # 一个句子的最大单词量
rnn_size = 10 # 相当于隐藏层的节点数
embedding_size = 50 # 一个单词的维数
min_word_frequency = 10 # 单词的最小词频
learning_rate = 0.0005
dropout_keep_prob = tf.placeholder(tf.float32) # 概率占位符

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
y_shuffled = np.array(text_data_target)[shuffled_ix] # 这个要注意下，list不能取list[1, 2, 3]

# 生成训练集与测试集
ix_cutoff = round(len(y_shuffled) * 0.8)
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_) # 获取字典的大小，单词数量为933
print("vocabulary size : {:d}".format(vocab_size))

# 声明计算图的占位符
x_data = tf.placeholder(shape=[None, max_sequence_length], dtype=tf.int32)
y_output = tf.placeholder(shape=[None], dtype=tf.int32)
# 构建词嵌入矩阵
embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], maxval=1.0, minval=-1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data) # embedding_output的shape=(?,25,50)

# 声明算法模型
####################
'''
两个######之间的算法为核心算法，时进行rnn的整个过程
1.一开始先得到句子的向量
2.通过句子的向量lookup词嵌入矩阵，我们可以得到词嵌入的结果，
                                shape=(batch_size, max_words{一个句子的最大长度}, embedding_size{词嵌入的向量维数})
3.利用tf.nn.dynamic_rnn()的方法我们可以得到整个过程中的输出值与状态值，
                                shape=(batch_size, max_words, rnn_size{rnn设置的节点数或者说是维数})
4.第3步所求的结果是整个过程的结果（还有进行dropout），我们的需求是只需要最后一个输出结果就行了，所以使用了transpose与gather这些函数
                                shape=(batch_size, rnn_size)
5.将这个结果带入输出层中进行y = w*x + b的计算过程，最后输出预测结果
6.当然这个过程还要进行softmax的过程
'''
if tf.__version__[0]>='1':
    cell=tf.contrib.rnn.BasicRNNCell(num_units = rnn_size)
else:
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units = rnn_size)
# cell = tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_size) # 这个初始化的cell可以看到output_size与state_size都是10
output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
output = tf.nn.dropout(output, dropout_keep_prob) # dropout是用来防止过拟合的

output = tf.transpose(output, [1, 0, 2]) # 这个变化矩阵的原因要注意
'''
last才是最后我们所需要的结果
'''
last = tf.gather(output, int(output.get_shape()[0]) - 1) # 这个写法就是用来取最后一个time_step的输出值
####################

# 为了完成rnn预测，我们通过全连接层将rnn_size大小的输出转换成二分类输出
weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[2]))
logits_out = tf.nn.softmax(tf.matmul(last, weight) + bias)

# 声明损失函数
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output)
loss = tf.reduce_mean(losses)

# 创建准确度函数比较训练集与测试集的训练结果
index = tf.argmax(logits_out, 1)
labels = tf.cast(y_output, tf.int64)
accuracy = tf.reduce_mean(tf.cast(tf.equal(index, labels), tf.float32))

optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)

# 训练
train_loss = []
test_loss = []
train_acc = []
test_acc = []
for epoch in range(epochs):
    writer = tf.summary.FileWriter('./rnn1', sess.graph)
    shuffled_ix = np.random.permutation(np.arange(len(x_train)))
    # x_train = x_train[shuffled_ix]
    # y_train = y_train[shuffled_ix]
    # feed_dict = {x_data : x_train, y_output : y_train}
    num_batches = int(len(x_train) / batch_size) + 1
    # TO DO CALCULATE GENERATIONS ExACTLY
    for i in range(num_batches):
        # Select train data
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i + 1) * batch_size)]) # 这个用法我之前在weibo_data处理中也想用
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]

        # Run train step
        train_dict = {x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob: 0.5}
        sess.run(train_step, feed_dict=train_dict)

    # Run loss and accuracy for training
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_acc.append(temp_train_acc)

    # Run Eval Step
    test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob: 1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_acc.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch + 1, temp_test_loss, temp_test_acc))

writer.close()

# Plot loss over time
epoch_seq = np.arange(1, epochs + 1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
plt.title('Softmax Loss')
plt.xlabel('Epochs')
plt.ylabel('Softmax Loss')
plt.legend(loc='upper left')
plt.show()

# Plot accuracy over time
plt.plot(epoch_seq, train_acc, 'k--', label='Train Set')
plt.plot(epoch_seq, test_acc, 'r-', label='Test Set')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()
