import random
import numpy as np
import datahandler.wsdream_handler as wh
from recommenders.irecommender import ItemBasedLSHRecommender
from recommenders.irecommender_old import ItemBasedLSHRecommenderOld
import json
from matplotlib import pyplot as plt

def test_predict_with_matrix(seed):
    org_data, data = wh.prepare_data(0.99, seed)
    (num_of_users, num_of_services) = data.shape
    test_samples = wh.prepare_test_data(num_of_users, 20, seed + 1)

    recommender = ItemBasedLSHRecommender(data, num_of_functions=4, seed=(seed+2))
    recommender.classify()

    degree = 7

    rmaes = np.zeros(degree)
    fails = np.zeros(degree)
    num_of_similar = np.zeros(degree)

    for t in range(degree):
        (rmaes[t], fails[t], num_of_similar[t]) = recommender.evaluate(data[test_samples], org_data[test_samples], t)

    return rmaes, fails, num_of_similar

def test_vary_with_threshold(times = 50, degree = 7):

    rmaes = np.zeros(degree)
    fails = np.zeros(degree)
    num_of_similar = np.zeros(degree)

    average_rmaes = np.zeros((times, degree))
    average_fails = np.zeros((times, degree))

    for t in range(times):
        (_rmaes, _fails, _num_of_similar) = test_predict_with_matrix(t + 1)
        rmaes += _rmaes
        fails += _fails
        num_of_similar += _num_of_similar
        # average_rmaes[t] = rmaes/(t + 1)
        # average_fails[t] = fails/(t + 1)
        if (t + 1) % 10 == 0:
            print('>')
        else:
            print('>', end='')

    dict = {}
    dict['rmaes'] = rmaes/times #average_rmaes.tolist()
    dict['fails'] = fails/times #average_fails.tolist()
    dict['num_of_similars'] = num_of_similar/times

    with open('../../outputs/irecommender/similar_test_50times.json', 'w') as f:
        json.dump(dict, f)

# def test_vary_with_hash_function(times)
def draw_vary_with_threshold():
    with open('../../outputs/irecommender/threshold_test_100times.json', 'r') as f:
        data = json.load(f)
        average_rmaes = np.array(data['average_rmaes'])

        # colors = []
        # for i in range(7):
        #     colors.append((0.1 * (i+1), 0.1 * (i+1), 0.1*(i+1)))
        # for i in range(7):
        #     plt.plot(range(100), average_rmaes[:, i], color=colors[i])

        plt.plot(range(4), average_rmaes[49, 0:4], color='b')
        plt.show()

        print(average_rmaes[48:49, :])

# draw_vary_with_threshold()
# //test()
test_vary_with_threshold()