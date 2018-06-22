
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

        #构建二进制矩阵
        dimensions = self.num_of_functions * shape[1]
        self.reduced_dimensions = int(np.ceil(np.log2(dimensions)))
        self.binary_matrix = np.zeros((dimensions, self.reduced_dimensions))
        for i in range(dimensions):
            str = np.binary_repr(i, width = self.reduced_dimensions)
            self.binary_matrix[i, :] = list(str)
        self.binary_matrix[self.binary_matrix == 0] = -1
        #构建转换为十进制的各位权重
        self.weights = [1 << x for x in range(self.reduced_dimensions)]

    def get_hash_value(self, x):
        '''
        compute the hash value of x
        :param x: (len(self.parameters), len(self.weights))
        :return:
        '''
        # 先计算点积
        values = np.dot(self.parameters, x)
        values = (values > 0).astype(float)
        # 再计算simhash
        v = values.flatten()
        values = np.sum(self.binary_matrix[v > 0], axis=0)
        values = (values > 0).astype(float)

        return np.dot(values, self.weights)
