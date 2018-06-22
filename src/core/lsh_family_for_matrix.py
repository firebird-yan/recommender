
import numpy as np

class LSHForMatrix:
    '''
    该类主要用于处理矩阵数据的LSH函数
    通常情况下，只有一个hash function
    '''
    def __init__(self, num_of_functions, seed = 1):
        self.num_of_functions = num_of_functions
        self.seed = seed

    def fit(self, shape):
        #这个可以保证每次随机生成的值相同
        np.random.seed(self.seed)
        self.parameters = np.zeros((self.num_of_functions, shape[0]))
        for i in range(self.num_of_functions):
            self.parameters[i, :] = np.random.uniform(-1, 1, size=shape[0])

        #构建转换为十进制的各位权重
        self.weights = [1 << x for x in range(self.num_of_functions)]
        self.weights = np.array(self.weights).reshape(1, self.num_of_functions)

    def get_hash_value(self, x):
        '''
        compute the hash value of x
        :param x: (len(self.parameters), len(self.weights))
        :return:
        '''
        # 先计算点积
        values = np.dot(self.parameters, x)
        values = (values > 0).astype(float)

        values = np.squeeze(np.dot(self.weights, values))
        return tuple(values[0:3])
