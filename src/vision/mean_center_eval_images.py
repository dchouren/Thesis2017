import h5py
import numpy as np

from vision_utils import mean_center

import ipdb



f = h5py.File('/tigress/dchouren/thesis/evaluation/pairs.h5')

with h5py.File('/tigress/dchouren/thesis/evaluation/pairs_mc.h5') as outf:

    old_pairs = f['pairs']
    old_categories = f['categories']
    # ipdb.set_trace()
    mean_centered = [mean_center(x * 255, 'channels_first') for x in old_pairs]
    pairs = outf.create_dataset('pairs', data=np.array(mean_centered))
    categories = outf.create_dataset('categories', data=np.array(old_categories))


f.close()