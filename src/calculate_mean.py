import sys
import glob
from os.path import join
import time

import numpy as np
import cv2


image_dir = sys.argv[1]

b_avgs = []
g_avgs = []
r_avgs = []

images = glob.glob(join(image_dir, '*', '*', '*.jpg'))
for i, image in enumerate(images):
    if i % 10000 == 0:
        print(i, time.ctime())

    im = cv2.imread(image,1)  # color
    b,g,r = cv2.split(im)
    b_avg = np.mean(b)
    g_avg = np.mean(g)
    r_avg = np.mean(r)

    b_avgs.append(b_avg)
    g_avgs.append(g_avg)
    r_avgs.append(r_avg)

final_b_avg = np.mean(b_avgs)
final_g_avg = np.mean(g_avgs)
final_r_avg = np.mean(r_avgs)

print(final_b_avg, final_g_avg, final_r_avg)
