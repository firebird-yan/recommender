import numpy as np
import datahandler.wsdream_handler as wh
from matplotlib import pyplot as plt
import json

def load_result():
    '''
    加载不同ratio下的mae矩阵（#hash_table * #hash_function)
    :return:一个#ratio * #hash_table * #hash_function的rmae和fail
    '''
    rmaes = []
    average_rames = []
    fails = []


    for ratio in range(7):
        file_name = '%s%d_result.json'%('../../outputs/irecommender/', ratio + 1)

        with open(file_name, 'r') as f:
            content = json.load(f)
            average_rames.append(content['rmae_of_average'])
            rmaes.append(content['rmaes'])
            fails.append(content['fails'])

    return np.array(rmaes), np.array(fails), np.array(average_rames)

def draw_lines():
    rmaes, _, _ = load_result()
    markers = ['.', 'o', '*', '1', '1', '2', '3']
    num_options = [4, 6, 8, 10]

    labels = []
    for i in num_options:
        for j in num_options:
            labels.append('%d*%d'%(i, j))


    for i in range(7):
        plt.plot(labels, rmaes[i].T.flatten(), marker=markers[i])

    plt.show()

def compare_fails(width = 5):
    _, fails, _ = load_result()
    num_options = [4, 6, 8, 10]

    labels = []
    for i in num_options:
        for j in num_options:
            labels.append('%d*%d' % (i, j))

    x = np.array(range(16)) + 1
    for i in range(7):
        plt.bar(x + width * i, fails[i].T.flatten(), label = 'ratio=%.1f'%((i + 1) * 0.1))
    plt.legend()

    plt.show()


# draw_lines()
compare_fails()