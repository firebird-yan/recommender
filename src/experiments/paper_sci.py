import random
import numpy as np
import datahandler.wsdream_handler as wh
from recommenders.irecommender import ItemBasedLSHRecommender
from recommenders.irecommender_old import ItemBasedLSHRecommenderOld
import json

def compute_mae_with_average_method(org_data, data, test_samples):
    """"
    基于提供的data，采用直接求每个用户所有可用rt值的平均值的预测方法，输出RMAE值
    :param org_data 原始数据
    :param data 随机将一定比例的数据进行置0处理后的数据
    :param test_samples: 测试用户的下标集
    :return rmae值
    """
    maes = []
    reference_columns = []

    for i in test_samples:
        user = data[i]
        columns = np.argwhere(user == 0)
        avg = np.average(user[user > 0])
        num_of_available = len(user[user > 0])
        reference_columns.append(num_of_available)
        for c in columns:
            if org_data[i][c] != -1:
                maes.append(np.abs(org_data[i][c] - avg))

    maes = np.array(maes)
    rmae = np.sqrt(np.dot(maes.T, maes) / maes.shape[0])

    return rmae

def compute_mae_with_lsh(org_data, data, test_samples, seed, num_of_hash_functions = 4, num_of_hash_tables = 8):
    '''
    计算采用LSH的方法计算用户的相似度的预测算法的rmae值
    :param org_data: 原始数据
    :param data: 经过按指定比例随机置零处理后的数据
    :param test_samples: 测试用户的下标集合
    :param seed: 随机种子
    :param num_of_hash_functions: lsh中hash_function的个数
    :param num_of_hash_tables: hash_table的个数
    :return:
    '''
    recommender = ItemBasedLSHRecommender(data, num_of_hash_functions, num_of_hash_functions, seed)
    recommender.classify()

    rmae, failed = recommender.evaluate(data[test_samples], org_data[test_samples])

    return rmae, failed

def tune_lsh_parameters(ratio, seed, times):
    '''
    该方法以平均值方法为参考，对比不同的#hash_function和#hash_table下mae值的变化
    以找到最优的#hash_function和#hash_table
    :param ration:
    :param seed:
    :return:
    '''
    hash_function_options = [4, 6, 8, 10]
    hash_table_options = [4, 6, 8, 10]
    rmae_of_average = 0
    rmaes = np.zeros((len(hash_table_options), len(hash_function_options)))
    fails = np.zeros(rmaes.shape)

    for i in range(times):
        _rmae_of_average, _rmaes, _fails = evaluate_vary_with_lsh_parameters(ratio, seed,
                                                        hash_function_options, hash_table_options)
        rmae_of_average += _rmae_of_average
        rmaes += _rmaes
        fails += _fails

        seed += 100
        if (i + 1) % 20 == 0:
            print('>')
        else:
            print('>', end = '')

    dict = {}
    dict['rmae_of_average'] = np.asscalar(rmae_of_average/times)
    dict['rmaes'] = (rmaes/times).tolist()
    dict['fails'] = (fails/times).tolist()
    file_name = '%s%d_result.json'%('E:/FangCloudSync/gtd/outputs/irecommender/', int(ratio*10))

    with open(file_name, 'w') as f:
        json.dump(dict, f)

    print('\n=======================ratio:', ratio, '=====================================')

def evaluate_vary_with_lsh_parameters(ratio, seed, hash_function_options, hash_table_options):
    '''
    测试不同#hash_function和#hash_table下的mae和fails值
    :param ratio:
    :param seed:
    :return:
    '''
    org_data, data = wh.prepare_data(ratio, seed)
    seed += 1
    (num_of_users, num_of_services)  = data.shape

    test_samples = wh.prepare_test_data(num_of_users, 50, seed)
    seed += 1

    rmae_of_average = compute_mae_with_average_method(org_data, data, test_samples)



    num_of_function_options = len(hash_function_options)
    num_of_table_options = len(hash_table_options)
    rmaes = np.zeros((num_of_table_options, num_of_function_options))
    fails = np.zeros((num_of_table_options, num_of_function_options))
    for i in range(num_of_table_options):
        for j in range(num_of_function_options):
            seed += 10
            rmaes[i][j],fails[i][j] = compute_mae_with_lsh(org_data, data, test_samples, seed,
                                                          hash_function_options[j], hash_table_options[i])

    return rmae_of_average, rmaes, fails

ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
seed = 1 #先运行一个ratio100次，运行下一个的时候需要seed=2，依次类推
for ratio in ratios:
    tune_lsh_parameters(ratio, seed, 1)
    seed += 1000