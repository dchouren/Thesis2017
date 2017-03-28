import sys

from PIL import Image
import requests
from itertools import zip_longest
import h5py
import numpy as np

import vision_utils as vutils
from keras.preprocessing import image

import ipdb



def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def download_url(url, img_size, rescale=False):
    im = Image.open(requests.get(url, stream=True).raw)
    im = im.resize(img_size)
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    if rescale:
        im *= 1./255

    return im


test_file = sys.argv[1]
output = sys.argv[2]

with open(test_file, 'r') as inf:
    test_urls = grouper(inf.readlines(), 4)


img_size = (224,224)

test_pairs = []
categories = []

for category, base_url, pos_url, neg_url in test_urls:
    print(category)
    base_image = download_url(base_url, img_size, rescale=True)
    pos_image = download_url(pos_url, img_size, rescale=True)
    neg_image = download_url(neg_url, img_size, rescale=True)

    test_pairs += [[base_image, pos_image]]
    test_pairs += [[base_image, neg_image]]

    categories += [category]

with h5py.File(output, 'w') as outf:
    outf.create_dataset('pairs', data=test_pairs)
    outf.create_dataset('categories', data=categories)





# im = Image.open(requests.get(url, stream=True).raw)

