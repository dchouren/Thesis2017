import sys

from PIL import Image
import requests
from itertools import zip_longest
import h5py
import numpy as np

import vision_utils as vutils
from keras.preprocessing import image

import ipdb


def grayscale_to_rbg(im):
    im = np.repeat(im, 3, 2)
    return im


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def download_url(url, img_size, rescale=False):
    im = Image.open(requests.get(url, stream=True).raw)
    im = im.resize(img_size)
    im = image.img_to_array(im)

    if im.shape[-1] != 3:
        im = grayscale_to_rbg(im)
    # im = np.expand_dims(im, axis=0)
    if rescale:
        im *= 1./255

    return np.array(im)


test_file = sys.argv[1]
output = sys.argv[2]

with open(test_file, 'r') as inf:
    test_urls = grouper(inf.readlines(), 4)


img_size = (224,224)

test_pairs = []
categories = []

for i, (category, base_url, pos_url, neg_url) in enumerate(test_urls):
    category = category.strip()
    print(i, category)
    base_image = download_url(base_url, img_size, rescale=True)
    pos_image = download_url(pos_url, img_size, rescale=True)
    neg_image = download_url(neg_url, img_size, rescale=True)

    # if i == 22:
    #     ipdb.set_trace()

    test_pairs += [[base_image, pos_image]]
    test_pairs += [[base_image, neg_image]]

    categories += [category.encode('utf8')] * 2

# ipdb.set_trace()

test_pairs = np.array(test_pairs)
categories = np.array(categories)

with h5py.File(output, 'w') as outf:
    outf.create_dataset('pairs', data=test_pairs)
    outf.create_dataset('categories', data=categories)





# im = Image.open(requests.get(url, stream=True).raw)












