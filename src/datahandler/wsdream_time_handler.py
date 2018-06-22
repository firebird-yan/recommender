import numpy as np
import json
import time
import random

def load_rt_demo():
    lines = []
    with open('../../datasets/WSDream-3-142-4532-64/dataset#2/rtdata.txt', 'r') as f:
        count = 0
        for line in f:
            lines.append(line)
            count += 1

            if count > 10000:
                break

    with open('../../datasets/WSDream-3-142-4532-64/dataset#2/rtdata_demo.txt', 'w') as f:
        f.writelines(lines)

def load_rt():
    '''
    加载response time data
    :return:
    '''
    with open('../../datasets/WSDream-3-142-4532-64/dataset#2/rtdata.txt', 'r') as f:
        num_users = 142
        num_services = 4532
        num_time_slices = 64
        data = np.zeros((num_users, num_services, num_time_slices))
        count = 0
        for line in f:
            values = line.split()
            user_id = int(values[0])
            service_id = int(values[1])
            time_id = int(values[2])
            data[user_id, service_id, time_id] = float(values[3])
            count += 1
            if count % 100000 == 0:
                print('>', end = '')

        with open('../../datasets/WSDream-3-142-4532-64/dataset#2/rtdata.json', 'w') as f:
            json.dump(data.tolist(), f)

def load_rt_from_json():
    begin = time.time()
    with open('../../datasets/WSDream-3-142-4532-64/dataset#2/rtdata.json', 'r') as f:
        data = json.load(f)
    print('load cost ', time.time() - begin, 's')
    return np.array(data)

def erase_data_randomly(data, ratio):
    '''
    将data中最后一个时间维度的数据随机抹零
    :param data:
    :param ratio:
    :return:
    '''
    processed_data = np.copy(data)
    if ratio < 1.0:
        total_length = data.size

        random_indices = random.sample(range(total_length),
                                       int(np.floor(total_length * ratio)))
        processed_data[:, :, 63].ravel()[random_indices] = 0

    return processed_data

