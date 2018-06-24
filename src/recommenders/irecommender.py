import numpy as np
from core.lsh_family import LSH
from recommenders.irecommender_old import ItemBasedLSHRecommenderOld
import datahandler.wsdream_handler as wh
import time
import random

class ItemBasedLSHRecommender:
    def __init__(self, data, num_of_functions = 4, num_of_tables = 8, seed = 1):
        '''
        :param data: shape (num_of_users, num_of_services, num_of_time_slices)
        :param ratio: erase ratio of data[:, :, 63], the erased element will be set to 0
        :param parameters: parameters of local sensitive hash function, in consistent with num_of_hash_tables
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

    def find_similar_services(self, index):
        '''
        find similar serivces of specified service y
        :param y: (num_of_users, num_of_time_slices)
        :return:
        '''
        v = np.squeeze(self.similarity_matrix[index])

        similar_services =  np.where(v > 0)
        return similar_services

    def evaluate(self, test_data, reference_data):
        '''
        预测test_data中值为0的response time值，并计算所有预测值的绝对差
        evaluate mae
        :param test_data:
        :param reference_data:
        :return:
        '''
        num_of_test = len(test_data)
        maes = []
        num_of_failed = 0

        for i in range(num_of_test):
            begin = time.time()
            columns = np.argwhere(test_data[i] == 0)

            for c in columns:
                if reference_data[i][c] != -1:
                    similar_services = self.find_similar_services(c)

                    similar_columns = test_data[i, similar_services]
                    similar_columns = similar_columns[similar_columns > 0]
                    if len(similar_columns) > 0:
                        maes.append(np.abs(np.average(similar_columns) - reference_data[i][c]))
                    else:
                        num_of_failed += 1

        maes = np.array(maes)
        rmae = np.sqrt(np.dot(maes.T, maes)/maes.shape[0])

        return rmae, num_of_failed


