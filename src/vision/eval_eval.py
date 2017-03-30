import sys
from os.path import join

import numpy as np

import ipdb


def rank_of_match(preds, index):
    our_embedding = preds[index]
    dist_to_index = [(i, np.linalg.norm(embedding - our_embedding)) for i, embedding in enumerate(preds)]

    dist_to_pos = dist_to_index[index+1][1]

    # ipdb.set_trace()

    sorted_dists = sorted(dist_to_index, key=lambda x: x[1])

    sorted_dist_to_index = sorted(dist_to_index, key=lambda x: x[1])

    rank = [x[0] for x in sorted_dist_to_index[::3]].index(index+1) - 1
    return rank



eval_file = sys.argv[1]

eval_dir = '/tigress/dchouren/thesis/evaluation'

eval_preds = np.load(join(eval_dir, eval_file))
eval_preds = eval_preds.ravel()

pair_preds = zip(eval_preds[::2], eval_preds[1::2])

accuracy = [1 if pair_pred[0] < pair_pred[1] else 0 for pair_pred in pair_preds]

# ranks = [rank_of_match(eval_preds, i) for i in range(0, len(eval_preds-3))]

# print(sum(ranks))

# ipdb.set_trace()
print(sum(accuracy) / len(accuracy))