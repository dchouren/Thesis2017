import sys
from os.path import join

from PIL import Image
import numpy as np
import h5py
import matplotlib.pyplot as plt

import ipdb


pairs_file = sys.argv[1]

sample_size = 20
rand_pairs = np.load(pairs_file)

# f = h5py.File(join('/tigress/dchouren/thesis/resources/pairs', pairs_file))
# pairs = f['pairs']

# rand_indices = np.random.choice(pairs.shape[0], sample_size)
# rand_pair_indices = []
# for rand_index in rand_indices:
#     if rand_index % 2 == 0:
#         rand_pair_indices += [rand_index, rand_index+1]
#     else:
#         rand_pair_indices += [rand_index-1, rand_index]

# rand_pair_indices = sorted(rand_pair_indices)

# rand_pairs = pairs[rand_pair_indices]

# np.save('/tigress/dchouren/thesis/resources/pairs/rand_pairs.npy', rand_pairs)
# ipdb.set_trace()
rand_pairs *= 255.
# rand_pairs[:, :, 0, :, :] += 94.78833771
# rand_pairs[:, :, 1, :, :] += 91.02941132
# rand_pairs[:, :, 2, :, :] += 88.50362396

# ipdb.set_trace()

plt.gcf().clear()
# plt.figure(figsize=(4,10))
num_pairs = 8
for i, (left, right) in enumerate(rand_pairs[:num_pairs]):
    # ipdb.set_trace()
    left = np.array(np.swapaxes(np.swapaxes(left, 0, 2), 0, 1), np.uint8)
    plt.subplot(num_pairs, 2, 2*i+1)
    plt.imshow(left)

    right = np.array(np.swapaxes(np.swapaxes(right, 0, 2), 0, 1), np.uint8)
    plt.subplot(num_pairs, 2, 2*i+2)
    plt.imshow(right)
# ipdb.set_trace()

# plt.subplots_adjust(right=2.0, top=2.0)
plt.axis('off')
plt.show()








