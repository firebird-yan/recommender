# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:56:49 2017
LSH函数族实现，函数族中hash value的位数由num_hash_function决定
使用前需先调用fit函数为每个hash位生成随机向量paramerters
每个hash位就是用parameters[i]与数据作点乘，如果>0,hash值为1，否则为0
@author: Yanchao
"""
import numpy as np
import random
class LSH:
    def __init__(self, num_of_functions):
        self.num_of_functions = num_of_functions
        self.parameters = None

    def fit(self, dimension, seed):
        # 这个可以保证每次随机生成的值相同
        np.random.seed(seed)
        self.parameters = np.zeros((self.num_of_functions, dimension))
        for i in range(self.num_of_functions):
            self.parameters[i, :] = np.random.uniform(-1, 1, size=dimension)

        # 构建转换为十进制的各位权重
        self.weights = [1 << x for x in range(self.num_of_functions)]
        self.weights = np.array(self.weights).reshape(1, self.num_of_functions)


    def get_hash_value(self, x):
        '''
        计算x的hash值
        得到一个1*num_hash_functions维的向量，将向量中值大于0的置为1，小于等于0的置0
        然后将该向量转换为十进制，如[1 0 1 0]转换为5
        :param x: (dimension, 1)
        :return:
        '''
        x.reshape(self.parameters.shape[1], 1)
        values = np.dot(self.parameters, x)
        values = (values > 0).astype(float)

        return np.asscalar(np.squeeze(np.dot(self.weights, values)))

    def get_batch_hash_value(self, X):
        '''
        以矩阵运算的方式同时计算所有向量的hash值
        :param X: (m, self.parameters.shape[1])
        :return:
        '''
        values = np.dot(self.parameters, X)
        values = (values > 0).astype(float)
        return np.squeeze(np.dot(self.weights, values))
        
        

