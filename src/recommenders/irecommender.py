import numpy as np
from core.lsh_family import LSH
from recommenders.irecommender_old import ItemBasedLSHRecommenderOld
import datahandler.wsdream_handler as wh
import time
import random
import json

class ItemBasedLSHRecommender:
    def __init__(self, data, num_of_functions=4, num_of_tables=8, seed=1):
        '''
        :param data: shape (num_of_users, num_of_services, num_of_time_slices)
        :param num_of_functions:
        :param num_of_tables:
        :param seed: 随机函数的种子
        '''
        self.data = data
        (self.num_of_users, self.num_of_services) = data.shape
        #initialize lsh tables
        self.num_of_tables = num_of_tables
        self.lsh_family = []

        for i in range(self.num_of_tables):
            self.lsh_family.append(LSH(num_of_functions))
            self.lsh_family[i].fit(self.num_of_users, seed)
            seed += 1

    def classify(self):
        '''
        compute lsh value of items, and put them into different buckets by its lsh value
        :return:
        '''
        self.similarity_matrix = np.zeros((self.num_of_services, self.num_of_services))
        for i in range(self.num_of_tables):
            bucket = {}
            hash_values = self.lsh_family[i].get_batch_hash_value(self.data)
            for j in range(self.num_of_services):
                self.similarity_matrix[j, :][hash_values == hash_values[j]] += 1


    def evaluate(self, test_data, reference_data, threshold=0):
        '''
        预测test_data中值为0的response time值，并计算所有预测值的绝对差
        evaluate mae
        :param test_data:
        :param reference_data:
        :param threshold: 找相似用户的阈值，默认为0， 与传统的LSH方法一致
        :return:
        '''
        begin = time.time()
        num_of_test = len(test_data)

        #需要预测的数据的矩阵，(num_of_test, num_of_service)， 如果需要预测，对应值为True，否则为False
        to_be_predicted = (test_data == 0) * (reference_data != -1)
        #为了便于计算，将元素置为-1的置为0
        test_data[test_data == -1] = 0
        #计算每个service的可用的相似service的个数，以方便下一步的求平均
        user_matrix = (test_data > 0).astype(float)
        similar_services = (self.similarity_matrix > threshold).astype(float)
        available_count = np.dot(user_matrix, similar_services)
        #available_count = 0的即为无法预测的元素
        #需要先去除不需要预测的元素
        to_be_predicted_available_count = available_count[to_be_predicted]
        num_of_failed = len(to_be_predicted_available_count[to_be_predicted_available_count == 0])
        #为了避免除以0的情况，我们将available_count = 0的重新设置为很小的值，这样相除得到的预测值也不会影响最终的mae
        available_count[available_count == 0] = 0.00000000001

        #计算预测值
        predict_values = np.dot(test_data, similar_services)
        #排除predict_values为0的情况
        predict_values[predict_values == 0] = np.iinfo(int).max

        predict_values = predict_values/available_count

        #找出每个用户的所有sevice中预测值最小的，推荐给用户，并计算其mae
        bias = np.zeros(num_of_test)
        num_of_similar = np.zeros(num_of_test)
        for i in range(num_of_test):
            row = predict_values[i][to_be_predicted[i]]
            reference_row = reference_data[i][to_be_predicted[i]]
            index = np.argmin(row)
            bias[i] = row[index] - reference_row[index]
            num_of_similar[i] = available_count[i][to_be_predicted[i]][index]

        rmae = np.sqrt(np.dot(bias, bias)/num_of_test)

        return rmae, num_of_failed, np.average(num_of_similar)





