import numpy as np
import random

def load_ws_dataset():
    '''
    加载ws-dream数据集中response time数据
    :return:
    '''
    rt_data = []
    with open('C:/Users/Yanchao/Documents/python/recommender/datasets/ws-dream/rtMatrix.txt', 'r') as f:
        for line in f:
            values = [float(x) for x in line.split()]
            rt_data.append(values)
    return rt_data

def preprocess(data, ratio, seed):
    '''
    随机抹取ratio比例的评分数据，并将数据分为训练样本和测试样本
    :param data:
    :param ratio:
    :return: data
    '''
    processed_data = np.copy(data)
    #为了提高LSH的准确性，将-1置为0
    processed_data[processed_data == -1] = 0
    random.seed(seed)
    random_indices = random.sample(range(data.size),
                                   int(np.floor(data.size * ratio)))
    processed_data.ravel()[random_indices] = 0

    return processed_data

def write_2ddata_into_file(data, filename):
    with open(filename, 'w') as f:
        (row, column) = data.shape
        for i in range(row):
            for j in range(column - 1):
                f.write('%.4f\t'%data[i][j])

            f.write('%.4f\n'%data[i][column - 1])

def load_2ddata_from_file(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f:
            row = [float(x) for x in line.split('\t')]
            data.append(row)

        return data


def prepare_data(ratio, seed):
    """
    对外公开的接口
    加载数据，并进行预处理，将ratio比例的数据置为0
    :param ratio:
    :param seed:
    :return:
    """
    org_data = load_ws_dataset()
    org_data = np.array(org_data)
    # org_data = org_data[:, 0:500]
    data = preprocess(org_data, ratio, seed)

    return org_data, data


def prepare_test_data(max_index, num_of_test, seed):
    """
    对外公开的接口
    随机生成num_of_test个索引
    :param max_index:
    :param num_of_test:
    :param seed:
    :return:
    """
    random.seed(seed)
    test_samples = random.sample(range(max_index), num_of_test)

    return test_samples
