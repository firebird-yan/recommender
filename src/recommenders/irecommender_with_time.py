
import numpy as np
from core.lsh_family_for_matrix import LSHForMatrix
import random
import datahandler.wsdream_time_handler as wh
import time

class ItemBasedLSHRecommenderWithTime:
    def __init__(self, data, num_of_functions = 4, num_of_tables = 8, seed = 1):
        '''
        :param data: shape (num_of_users, num_of_services, num_of_time_slices)
        :param ratio: erase ratio of data[:, :, 63], the erased element will be set to 0
        :param parameters: parameters of local sensitive hash function, in consistent with num_of_hash_tables
        '''
        self.data = data
        (self.num_of_users, self.num_of_services, self.num_of_time_slices) = data.shape
        #initialize lsh tables
        self.num_of_tables = num_of_tables
        self.lsh_family = []

        for i in range(self.num_of_tables):
            self.lsh_family.append(LSHForMatrix(num_of_functions, seed))
            self.lsh_family[i].fit((self.num_of_users, self.num_of_time_slices))
            seed += 1

    def classify(self):
        '''
        compute lsh value of items, and put them into different buckets by its lsh value
        :return:
        '''
        self.buckets = []
        for i in range(self.num_of_tables):
            bucket = {}
            for j in range(self.num_of_services):
                hash_value = self.lsh_family[i].get_hash_value(self.data[:, j, :])
                if bucket.get(hash_value) is None:
                    bucket[hash_value] = []
                bucket[hash_value].append(j)
            self.buckets.append(bucket)

    def find_similar_services(self, y):
        '''
        find similar serivces of specified service y
        :param y: (num_of_users, num_of_time_slices)
        :return:
        '''
        similar_services = []
        for i in range(self.num_of_tables):
            hash_value = self.lsh_family[i].get_hash_value(y)

            if self.buckets[i].get(hash_value) is not None:
                similar_services.extend(self.buckets[i][hash_value])

        return list(set(similar_services))

    def evaluate(self, test_data, reference_data):
        '''
        预测test_data中值为0的response time值，并计算所有预测值的绝对差
        evaluate mae
        :param test_data:
        :param reference_data:
        :return:
        '''

def test_num_of_users():
    '''
    test the number of similar_services under different num_of_hash_table
    :return:
    '''
    data = wh.load_rt_from_json()

    data = data[:, :-8, :]  #最后8个service的数据全为0
    (num_of_users, num_of_services, _) = data.shape
    test_samples = random.sample(range(num_of_services), 500)


    hash_table_options = [4, 6, 8, 10, 12]
    hash_function_options = [2, 4, 6, 8, 10, 12]
    num_table_options = 5
    num_function_options = 4
    num_of_similar_services = np.zeros((num_table_options, num_function_options))
    num_of_isolated_services = np.zeros((num_table_options, num_function_options))
    for i in range(num_table_options):
        for j in range(num_function_options):
            num_of_hash_table = hash_table_options[i]
            num_of_hash_function = hash_function_options[j]
            begin = time.time()
            recommender = ItemBasedLSHRecommenderWithTime(data, num_of_hash_function, num_of_hash_table, 2)
            recommender.classify()
            print('num_of_hash_table = %d, prepare cost %.4f'%(num_of_hash_table, time.time()-begin))
            begin = time.time()
            nums = 0
            isolated_count = 0
            for t in test_samples:
                similar_services = recommender.find_similar_services(data[:, t, :])
                num = len(similar_services)
                if num == 0:
                    isolated_count += 1

                nums += num
            num_of_similar_services[i][j] = nums/500.0
            num_of_isolated_services[i][j] = isolated_count
            print('num_of_hash_table = %d, find similar services cost %.4fs'%(num_of_hash_table, time.time() - begin))

    print(num_of_similar_services)
    print(num_of_isolated_services)

test_num_of_users()



