import sys
from os.path import join

import cv2
from src.vision.quality_utils import get_laplacian
import numpy as np
import glob

import matplotlib.pyplot as plt

import ipdb

scores = []

filename = sys.argv[1]
im_dir = sys.argv[2]


def get_blurriness(laplacian):
    laplacian = laplacian.flatten()
    sharp_pixels = len(np.where(laplacian > 128)[0]) + len(np.where(laplacian < -128)[0])
    blurriness = sharp_pixels / len(laplacian) * 100
    return blurriness

# with open(filename, 'r') as inf:
#     new_lines = []
#     for line in inf.readlines():
#
#         tokens = line.strip().split(',')
#         url = tokens[0]
#         im_name = url.split('photo-s/')[-1].replace('/','-')
#         im_path = join(im_dir, im_name)
#
#         cv_im = cv2.imread(im_path, 0)
#         laplacian = get_laplacian(cv_im)
#         if laplacian == None:
#             continue
#         blurriness = get_blurriness(laplacian)
#
#         tokens = tokens[:-2] + [blurriness] + tokens[-2:]
#         new_lines.append(tokens)
#         scores.append(blurriness)
#
#
# with open(filename + '_laplacian', 'w') as outf:
#     for new_line in new_lines:
#         outf.write(','.join([str(x) for x in new_line]) + '\n')

for f in glob.glob('./*.jpg'):
    print(f)
    cv_im = cv2.imread(f, 0)
    score = get_blurriness(get_laplacian(cv_im))
    print(score)

ipdb.set_trace()

