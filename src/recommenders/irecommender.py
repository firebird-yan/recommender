import numpy as np
from core.lsh_family import LSH
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
        v = self.similarity_matrix[index, :]

        return np.argwhere(v > 0)

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
        reference_columns = []

        for i in range(num_of_test):
            begin = time.time()
            columns = np.argwhere(test_data[i] == 0)

            for c in columns:
                if reference_data[i][c] != -1:
                    similar_services = self.find_similar_services(c)
                    similar_columns = test_data[i][similar_services]
                    similar_columns = similar_columns[similar_columns > 0]
                    if len(similar_columns) > 0:
                        maes.append(np.abs(np.average(similar_columns) - reference_data[i][c]))
                        reference_columns.append(len(similar_columns))
                    else:
                        # print('no available similar services! %d'%(len(similar_services)))
                        num_of_failed += 1

        maes = np.array(maes)
        rmae = np.sqrt(np.dot(maes.T, maes)/maes.shape[0])
        return rmae, num_of_failed, reference_columns


def prepare_data(ratio, seed):
    """
    加载数据，并进行预处理，将ratio比例的数据置为0
    :param ratio:
    :param seed:
    :return:
    """
    org_data = wh.load_ws_dataset()
    org_data = np.array(org_data)
    # org_data = org_data[:, 0:500]
    data = wh.preprocess(org_data, ratio, seed)

    return org_data, data


def prepare_test_data(max_index, num_of_test, seed):
    """
    随机生成num_of_test个索引
    :param max_index:
    :param num_of_test:
    :param seed:
    :return:
    """
    random.seed(seed)
    test_samples = random.sample(range(max_index), num_of_test)

    return test_samples


def test_num_of_simliar_users(ratio, seed):
    """
    测试不同的num_of_hash_table 和 num_of_hash_functions下返回的相似用户个数
    :param ratio:
    :param seed:
    :return:
    """
    data, org_data = prepare_data(ratio, seed)
    (num_of_users, num_of_services) = data.shape
    test_samples = prepare_test_data(num_of_services, 50, seed)


    hash_table_options = [4, 6, 8, 10, 12]
    hash_function_options = [4, 6, 8, 10, 12]
    num_table_options = 1
    num_function_options = 5
    num_of_similar_services = np.zeros((num_table_options, num_function_options))
    num_of_isolated_services = np.zeros((num_table_options, num_function_options))
    for i in range(num_table_options):
        for j in range(num_function_options):
            num_of_hash_table = hash_table_options[i]
            num_of_hash_function = hash_function_options[j]
            begin = time.time()
            recommender = ItemBasedLSHRecommender(data, num_of_hash_function, num_of_hash_table)
            recommender.classify()
            # print('num_of_hash_table = %d, prepare cost %.4f' % (num_of_hash_table, time.time() - begin))
            begin = time.time()
            nums = 0
            isolated_count = 0
            for t in test_samples:
                similar_services = recommender.find_similar_services(t)
                num = len(similar_services)
                if num == 0:
                    isolated_count += 1

                nums += num
            num_of_similar_services[i][j] = nums / 500.0
            num_of_isolated_services[i][j] = isolated_count
            # print('num_of_hash_table = %d, find similar services cost %.4fs' % (num_of_hash_table, time.time() - begin))

    print(num_of_similar_services)
    print(num_of_isolated_services)

def mae_of_average(ratio, seed):
    """
    该方法测试采用平均值的方法得到的mae值，以验证lsh方法的有效性
    :param ratio:
    :param seed:
    :return:
    """
    org_data, data = prepare_data(ratio, seed)
    (num_of_users, num_of_services) = data.shape

    test_samples = prepare_test_data(num_of_users, 50, seed + 1) #保证算法的一次迭代中用到的随机都是不同的

    maes = []
    reference_columns = []

    for i in test_samples:
        user = data[i]
        columns = np.argwhere(user == 0)
        avg = np.average(user[user > 0])
        num_of_available = len(user[user > 0])
        for c in columns:
            if org_data[i][c] != -1:
                maes.append(np.abs(org_data[i][c] - avg))
                reference_columns.append(num_of_available)

    maes = np.array(maes)
    rmae = np.sqrt(np.dot(maes.T, maes)/maes.shape[0])

    return rmae, reference_columns

def test_mae_of_lsh(ratio, seed, num_of_hash_functions = 4, num_of_hash_tables = 8):
    org_data, data = prepare_data(ratio, seed)
    (num_of_users, num_of_services) = data.shape
    test_samples = prepare_test_data(num_of_users, 50, seed + 1)

    recommender = ItemBasedLSHRecommender(data, num_of_hash_functions, num_of_hash_functions, seed + 2)
    recommender.classify()

    rmae, failed, reference_columns = recommender.evaluate(data[test_samples], org_data[test_samples])

    return rmae, failed, reference_columns


def test_mae(ratio, seed):
    org_data, data = prepare_data(ratio, seed)
    (num_of_users, num_of_services) = data.shape
    test_samples = prepare_test_data(num_of_users, 50, seed + 1)

    hash_table_options = [4, 6, 8, 10, 12]
    hash_function_options = [4, 6, 8, 10]
    num_table_options = 1
    num_function_options = 4
    maes = np.zeros((num_table_options, num_function_options))
    failed = np.zeros((num_table_options, num_function_options))

    for i in range(num_table_options):
        for j in range(num_function_options):
            # begin = time.time()
            recommender = ItemBasedLSHRecommender(data, hash_function_options[j], hash_table_options[i], seed)
            recommender.classify()
            # print('prepare cost ', time.time() - begin)

            # begin = time.time()
            maes[i][j], failed[i][j] = recommender.evaluate(data[test_samples], org_data[test_samples])
            # print('evaluate cost ', time.time() - begin)
            print('>', end='')
    print(maes)
    print(failed)
    dir = '../../outputs/irecommender/'
    # wh.write_2ddata_into_file(maes, dir+'maes_threshold_0.csv')
    # wh.write_2ddata_into_file(failed, dir + 'failed_threshold_0.csv')

# mae_of_average(0.2, 2)
def tune_ratio_parameters(times):
    ratios = [0.5, 0.7, 0.9, 0.98]
    rmaes_of_average = np.zeros((4, times))
    rmaes_of_lsh = np.zeros((4, times))

    for i in range(4):
        ratio = ratios[i]
        seed = 1
        for t in range(times):
            rmaes_of_average[i][t], refrence_columns1 = mae_of_average(ratio, seed + t)
            rmaes_of_lsh[i][t],_ , refrence_columns2 = test_mae_of_lsh(ratio, seed + t)

            print(np.average(refrence_columns1))
            print(np.average(refrence_columns2))

    print(np.average(rmaes_of_average, axis = 1))
    print(np.average(rmaes_of_lsh, axis = 1))
    # dir = '../../outputs/irecommender/'
    # wh.write_2ddata_into_file(rmaes_of_average, dir+'rmaes_of_average.csv')
    # wh.write_2ddata_into_file(rmaes_of_lsh, dir + 'rmaes_of_lsh.csv')

tune_ratio_parameters(1)