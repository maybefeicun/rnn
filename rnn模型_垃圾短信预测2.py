# -*- coding: utf-8 -*-
# @Time    : 2018/3/20 16:05
# @Author  : chen
# @File    : rnn模型_垃圾短信预测2.py
# @Software: PyCharm
import numpy as np

list_1 = [1, 2, 3, 4, 5, 6]
index = np.random.permutation(np.arange(5))
list_2 = np.array(list_1)[index]

list_3 = list_1[1]


str = 'wo men hha \n women wo men'
words = set(str)
pass