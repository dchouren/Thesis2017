import kd_tree
import numpy as np
import pickle
import time

import sys

import ipdb


class _KDTree(object):
    def __init__(self, data):
        self.index_data = [[float(x[0]), float(x[1])] for x in data]
        self.data = data
        self.kdtree = kd_tree.KDTree(self.index_data)

    def get_pos_pairs(self, r):
        pair_indexes = self.kdtree.query_pairs(r)
        all_pairs = [[self.data[pair_index[0]], self.data[pair_index[1]]] for pair_index in pair_indexes]

        return all_pairs

    def get_neg_pairs(self, r):
        pair_indexes = self.kdtree.query_negative_pairs(r)
        all_pairs = [[self.data[pair_index[0]], self.data[pair_index[1]]] for pair_index in pair_indexes]

        return all_pairs


if __name__ == '__main__':
    sys.setrecursionlimit(10000)

    data_file = sys.argv[1]
    sample_size = int(sys.argv[2])

    with open(data_file, 'r') as inf:
        data = inf.readlines()
    data = np.asarray([[float(x.split(',')[0]), float(x.split(',')[1]), *x.split(',')[2:]] for x in data])

    print(len(data))

    sample_data = data[np.random.choice(data.shape[0], sample_size)]
    K = _KDTree(sample_data)
    
    start_time = time.time()

    dist = 0.000014
    neg_dist = 0.1838
    pos_pairs = K.get_pos_pairs(dist)

    ipdb.set_trace()

    print(len(pos_pairs))

    # neg_pairs = K.get_neg_pairs(neg_dist)
    # np.save('./{}_test_pairs.npy'.format(sample_size), np.asarray(pos_pairs))
    # np.save('./{}_neg_test_pairs.npy'.format(sample_size), np.asarray(neg_pairs))
    
    # print('{}: {}'.format(sample_size, time.time() - start_time))
