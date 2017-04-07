import vision_utils as vutils

import time

im_path = '/scratch/network/dchouren/images/2012/01/8331121208_daaf6fa41c.jpg'
start_time = time.time()
im = vutils.load_and_preprocess_image(im_path, dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)



