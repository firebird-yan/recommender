
'''
本文件定义了一个user-based recommender算法类
相似用户基于LSH算法获得
'''
from core.lsh_family import LSH
import numpy as np

class UserBasedLSHRecommender:
    def __init__(self, train_data, num_hash_functions = 10, num_hash_tables = 8):
        self.train_data = train_data
        self.num_hash_functions = num_hash_functions
        self.num_hash_tables = num_hash_tables
        #initialize lsh_family
        self.lsh_family = []
        for i in range(num_hash_tables):
            self.lsh_family.append(LSH(num_hash_functions))
            #initialize the parameters of hash functions
            self.lsh_family[i].fit(self.train_data.shape[1])

    def train(self):
        '''
        计算train_data中所有样本的hash值（每个hash_table中都要计算一次）
        根据其hash值放入不同的buckets中
        :return:
        '''
        self.buckets = []
        m = len(self.train_data)

        for i in range(self.num_hash_tables):
            bucket = {}
            for j in range(m):
                hash_value = self.lsh_family[i].get_hash_value(self.train_data[j])

                if bucket.get(hash_value) is None:
                    bucket[hash_value] = []

                bucket[hash_value].append(j)

            self.buckets.append(bucket)

    def find_similar_users(self, x):
        '''
        根据某个用户的评分（x）查找x的相似用户
        :param x:
        :return:
        '''
        similar_users = []
        for i in range(self.num_hash_tables):
            hash_value = self.lsh_family[i].get_hash_value(x)

            if self.buckets[i].get(hash_value) is not None:
                similar_users.extend(self.buckets[i][hash_value])

        return list(set(similar_users))

    def evaluate(self, test_data, reference_data):
        '''
        评估算法的mae值
        :param test_data: array of test samples, with some scores masked
        :param reference_data: array of original data of test samples
        :return: average of mae
        '''
        m = len(test_data)
        n = len(test_data[0])
        mae = 0.
        total = 0
        for i in range(m):
            #predict the ith test user's score
            similar_users = self.find_similar_users(test_data[i])
            similar_data = self.train_data[similar_users]
            for j in range(n):
                if test_data[i][j] == 0 and reference_data[i][j] != -1:
                    valid_data = similar_data[similar_data[:,j] > 0]
                    if len(valid_data > 0):
                        predicted_value = np.average(valid_data[:, j])
                    else:
                        valid_data = test_data[i]
                        valid_data = valid_data[valid_data > 0]
                        if len(valid_data) > 0:
                            predicted_value = np.average(valid_data)
                        else:
                            print('no valid row data')
                            predicted_value = 0

                    mae += np.abs(predicted_value - reference_data[i][j])
                    total += 1
        print('total = ', total, ', mae = ', mae)
        return mae/total
