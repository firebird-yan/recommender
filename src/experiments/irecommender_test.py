import random
import numpy as np
import datahandler.wsdream_handler as wh
from recommenders.irecommender import ItemBasedLSHRecommender
from recommenders.irecommender_old import ItemBasedLSHRecommenderOld



def test_num_of_simliar_users(ratio, seed):
    """
    测试不同的num_of_hash_table 和 num_of_hash_functions下返回的相似用户个数
    :param ratio:
    :param seed:
    :return:
    """
    data, org_data = wh.prepare_data(ratio, seed)
    (num_of_users, num_of_services) = data.shape
    test_samples = wh.prepare_test_data(num_of_services, 50, seed)


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
    org_data, data = wh.prepare_data(ratio, seed)
    (num_of_users, num_of_services) = data.shape

    test_samples = wh.prepare_test_data(num_of_users, 50, seed + 1) #保证算法的一次迭代中用到的随机都是不同的

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
    org_data, data = wh.prepare_data(ratio, seed)
    (num_of_users, num_of_services) = data.shape
    test_samples = wh.prepare_test_data(num_of_users, 50, seed + 1)

    recommender = ItemBasedLSHRecommender(data, num_of_hash_functions, num_of_hash_functions, seed + 2)
    recommender.classify()

    rmae, failed, reference_columns = recommender.evaluate(data[test_samples], org_data[test_samples])

    return rmae, failed, reference_columns


def test_mae(ratio, seed):
    org_data, data = wh.prepare_data(ratio, seed)
    (num_of_users, num_of_services) = data.shape
    test_samples = wh.prepare_test_data(num_of_users, 50, seed + 1)

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

def tune_lsh_parameters(ratio, seed):
    '''
    该方法以平均值方法为参考，对比不同的#hash_function和#hash_table下mae值的变化
    以找到最优的#hash_function和#hash_table
    :param ration:
    :param seed:
    :return:
    '''
    org_data, data = wh.prepare_data(ratio, seed)
    (num_of_users, num_of_services)  = data.shape
    test_samples = wh.prepare_test_data(num_of_users, 50, seed + 1)

    rmae_of_average = compute_mae_with_average_method(org_data, data, test_samples)

    hash_function_options = [2, 4, 6, 8, 10]
    hash_table_options = [2, 4, 6, 8, 10]

    num_of_function_options = 5
    num_of_table_options = 5
    rmaes = np.zeros((num_of_table_options, num_of_function_options))
    for i in range(num_of_table_options):
        for j in range(num_of_function_options):
            seed += 1
            rmaes[i][j] = compute_mae_with_lsh(org_data, data, test_samples, seed,
                                                          hash_function_options[j], hash_table_options[i])

    print('========================ratio:', ratio, '===================')
    print('rmae of average:', rmae_of_average)
    print('rmae matrix:')
    print(rmaes)


def compute_mae_with_average_method(org_data, data, test_samples):
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
    recommender = ItemBasedLSHRecommender(data, num_of_hash_functions, num_of_hash_functions, seed)
    recommender.classify()

    rmae, _, reference_columns = recommender.evaluate(data[test_samples], org_data[test_samples])

    return rmae

def compute_mae_with_old_lsh(org_data, data, test_samples, seed, num_of_hash_functions = 4, num_of_hash_table = 8):
    recommender = ItemBasedLSHRecommenderOld(data, num_of_hash_functions, num_of_hash_functions, seed)
    recommender.classify()

    rmae, _, reference_columns = recommender.evaluate(data[test_samples], org_data[test_samples])


    return rmae

def compare_maes_with_old_lsh(ratio, seed):
    '''
    对比当前的lsh方法和之前的lsh方法的mae值
    以验证当前的lsh方法的正确性
    :param ratio:
    :return:
    '''
    org_data, data = wh.prepare_data(ratio, seed)
    (num_of_users, num_of_services) = data.shape
    test_samples = wh.prepare_test_data(num_of_users, 50, seed + 1)

    rmae_of_average = compute_mae_with_average_method(org_data, data, test_samples)

    hash_function_options = [4, 6, 8]
    hash_table_options = [4, 6, 8]

    num_of_function_options = 1
    num_of_table_options = 1
    rmaes = np.zeros((num_of_table_options, num_of_function_options))
    rmaes_old = np.zeros((num_of_table_options, num_of_function_options))
    for i in range(num_of_table_options):
        for j in range(num_of_function_options):
            seed += 1
            rmaes[i][j] = compute_mae_with_lsh(org_data, data, test_samples, seed,
                                               hash_function_options[j], hash_table_options[i])
            rmaes_old[i][j] = compute_mae_with_old_lsh(org_data, data, test_samples, seed,
                                               hash_function_options[j], hash_table_options[i])

    print('========================ratio:', ratio, '===================')
    print('rmae of average:', rmae_of_average)
    print('rmae matrix:')
    print(rmaes)
    print('rmae old matrix:')
    print(rmaes_old)

def compare_similar_users_with_old_lsh(ratio, seed):
    '''
    对比当前的lsh方法和之前的lsh方法的mae值
    以验证当前的lsh方法的正确性
    :param ratio:
    :return:
    '''
    org_data, data = wh.prepare_data(ratio, seed)
    (num_of_users, num_of_services) = data.shape
    test_samples = wh.prepare_test_data(num_of_users, 1, seed + 1)

    recommender_old = ItemBasedLSHRecommenderOld(data, 4, 4, seed)
    recommender_old.classify()

    recommender = ItemBasedLSHRecommender(data, 4, 4, seed)
    recommender.classify()

    num_of_similar_with_lsh = []
    num_of_similar_with_lsh_old = []
    for j in range(2):
        # num_of_similar_with_lsh.append(len(recommender.find_similar_services(j)))
        # num_of_similar_with_lsh_old.append(len(recommender_old.find_similar_services(data[:, j])))
        print(recommender.find_similar_services(j))
        print(recommender_old.find_similar_services(data[:,j]))

    print('lsh:', num_of_similar_with_lsh)
    print('lsh_old:', num_of_similar_with_lsh_old)

# compare_similar_users_with_old_lsh(0.5, 1)
# compare_maes_with_old_lsh(0.5, 1)
ratios = [0.2, 0.5, 0.7, 0.9]
seed = 1
for ratio in ratios:
    tune_lsh_parameters(ratio, seed)
    seed += 1