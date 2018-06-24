import numpy as np
import datahandler.wsdream_handler as wh
from matplotlib import pyplot as plt

def load_result():
    '''
    加载不同ratio下的mae矩阵（#hash_table * #hash_function)
    :return:一个#ratio * #hash_table * #hash_function的rmae和fail
    '''
    ratios = [2, 5, 7, 9]
    rmaes = []
    fails = []

    for ratio in ratios:
        rmae_file_name = '%s%d_rames.csv'%('../../outputs/irecommender/', ratio)
        fail_file_name = '%s%d_fails.csv'%('../../outputs/irecommender/', ratio)
        rmaes.append(wh.load_2ddata_from_file(rmae_file_name))
        fails.append(wh.load_2ddata_from_file(fail_file_name))

    return np.array(rmaes), np.array(fails)

def draw_lines():
    rmaes, _ = load_result()
    markers = ['.', 'o', '*', '1']
    num_options = [2, 4, 6, 8, 10]

    labels = []
    for i in num_options:
        for j in num_options:
            labels.append('%d * %d'%(i, j))


    for i in range(4):
        plt.plot(labels, rmaes[i].T.flatten(), marker=markers[i])

    plt.show()

draw_lines()