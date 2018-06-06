
import numpy as np
import random
from recommenders.urecommender import UserBasedLSHRecommender
import time

def load_ws_dataset():
    '''
    加载ws-dream数据集中response time数据
    :return:
    '''
    rt_data = []
    with open('../../datasets/ws-dream/rtMatrix.txt', 'r') as f:
        for line in f:
            values = [float(x) for x in line.split()]
            rt_data.append(values)
    return rt_data

def preprocess(data, ratio, num_train_examples):
    '''
    随机抹取ratio比例的评分数据，并将数据分为训练样本和测试样本
    :param data:
    :param ratio:
    :param num_train_examples:
    :return: train_data, test_data
    '''
    processed_data = np.copy(data)
    random_indices = random.sample(range(data.size),
                                   int(np.floor(data.size * ratio)))
    processed_data.ravel()[random_indices] = 0

    return processed_data[0:num_train_examples], processed_data[num_train_examples:]

def test_user_based_recommender():
    begin = time.time()
    rt_data = load_ws_dataset()
    rt_data = np.array(rt_data)
    num_train_examples = 300
    train_data, test_data = preprocess(rt_data, 0.3, num_train_examples)
    print('======data process cost ', time.time() - begin)

    num_hash_functions = [6, 8, 10]
    num_hash_tables = [6, 8, 10]
    for nt in num_hash_tables:
        for nf in num_hash_functions:
            begin = time.time()
            uRecommender = UserBasedLSHRecommender(train_data)
            uRecommender.train()
            # print('=======train process cost ', time.time() - begin)
            begin = time.time()
            mae = uRecommender.evaluate(test_data, rt_data[num_train_examples:])
            # print('test cost ', time.time() - begin, ', mae = ', mae)
            print('num_hash_functions = ', nf, ', num_hash_tables = ', nt, ', mae = ', mae)

test_user_based_recommender()

