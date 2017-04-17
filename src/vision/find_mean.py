import sys

import h5py
import numpy as np

import ipdb


def _generator(filename, batch_size=32, index=0):
    f = h5py.File(filename, 'r')
    pairs = f['pairs']
    while 1:
        yield pairs[index:index+batch_size]
        index += batch_size
        index = index % len(pairs)

    f.close()


filename = '/tigress/dchouren/thesis/resources/pairs/2015_all.h5'
batch_size = 3200
generator = _generator(filename, batch_size=batch_size, index=0)


all_channel_values = []
for i in range(0,int(713600 / batch_size)):
    print(i)
    batch = next(generator)
    # print(batch.shape)
    channel_avg = 255 * np.mean(batch, axis=(0,1,3,4))

    all_channel_values += [channel_avg]

np.save('/tigress/dchouren/thesis/resources/pairs/2015_channel_values.npy', np.array(all_channel_values))

print(np.mean(np.array(all_channel_values), axis=0))
# ipdb.set_trace()
