import numpy as np
import h5py
from os.path import join
import sys

from keras.preprocessing.image import ImageDataGenerator


import ipdb

def _fold_generator(filename, leave_out_start, leave_out_end, batch_size=32, index=0, augment=False):

    f = h5py.File(filename, 'r')
    pairs = f['pairs']
    total_pairs = pairs.shape[0]
    idg = ImageDataGenerator(rotation_range=20, horizontal_flip=True, vertical_flip=True, zoom_range=0.05)
    while 1:
        if index == leave_out_start:
            index = leave_out_end

        if index + batch_size > total_pairs:
            data = pairs[index:]
            index = 0
        else:
            data = pairs[index:index+batch_size]
            index += batch_size

        if augment:
            data = np.array([[idg.random_transform(x[0]), idg.random_transform(x[1])] for x in data])
        yield [data[:,0], data[:,1]], [1,0]*int(len(data)/2)

    f.close()


# num_pairs = 10066
# k = 10
# batch_size = 16
# augment = True

# skip = int((num_pairs - num_pairs % k*batch_size) / (k*batch_size)) * batch_size
# leave_out_indices = list(np.arange(0, num_pairs, skip))
# leave_out_indices.pop(-1)
# leave_out_indices += [0]
# leave_out_indices = np.array(leave_out_indices)

# print(leave_out_indices)

# pairs_file = '/tigress/dchouren/thesis/evaluation/pairs.h5'

# i = 7
# generator = _fold_generator(pairs_file, leave_out_indices[i], leave_out_indices[i+1], batch_size=batch_size, index=0, augment=augment)

# n_train_batch = int(leave_out_indices[i] / batch_size) + int((num_pairs - leave_out_indices[i+1]) / batch_size) + 1

# for k in range(0,10):
#     for j in range(n_train_batch):
#         gen = next(generator)
#         print(j, gen[0][0].shape, gen[0][1].shape, len(gen[1]))
#         assert gen[0][0].shape == gen[0][1].shape
#         assert gen[0][0].shape[0] == len(gen[1])
#         # assert gen[0][1].shape == (16,3,224,224)
#         # assert len(gen[1]) == 16
#     print('Finished {}'.format(k))

# ipdb.set_trace()



def _generator(filename, batch_size=32, index=0, augment=False):
    f = h5py.File(filename, 'r')
    pairs = f['pairs']
    total_pairs = pairs.shape[0]

    idg = ImageDataGenerator(rotation_range=20, horizontal_flip=True, vertical_flip=True, zoom_range=0.05)
    while 1:
        if index + batch_size > total_pairs:
            data = pairs[index:]
            index = 0
        else:
            data = pairs[index:index+batch_size]
            index += batch_size

        if augment:
            data = np.array([[idg.random_transform(x[0]), idg.random_transform(x[1])] for x in data])
        yield [data[:,0], data[:,1]], [1,0]*int(len(data[:,0])/2)
    f.close()


pairs_file = sys.argv[1]

batch_size = 32
pairs_file = join('/tigress/dchouren/thesis/resources/pairs/', pairs_file)
f = h5py.File(pairs_file)
x = f['pairs']
n_batch = int(x.shape[0] / batch_size)
f.close()

n_train_batch = int(n_batch / 10 * 9)
validation_steps = n_batch-n_train_batch

augment = True
generator = _generator(pairs_file, batch_size=batch_size, augment=augment)
val_generator = _generator(pairs_file, batch_size=batch_size, index=n_train_batch*batch_size, augment=augment)


for j in range(validation_steps):
    gen = next(generator)
    print(j, gen[0][0].shape, gen[0][1].shape, len(gen[1]))
    assert gen[0][0].shape == gen[0][1].shape
    assert gen[0][0].shape[0] == len(gen[1])
    # assert gen[0][1].shape == (16,3,224,224)
    # assert len(gen[1]) == 16
# print('Finished {}'.format(k))



