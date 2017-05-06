
import sys
from os.path import join
import time

import numpy as np
from skimage.transform import resize
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
# from tsne import bh_sne
import h5py
import cv2

import ipdb



def gray_to_color(img):
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))
    return img

def min_resize(img, size):
    """
    Resize an image so that it is size along the minimum spatial dimension.
    """
    w, h = map(float, img.shape[:2])
    if min([w, h]) != size:
        if w <= h:
            img = resize(img, (int(round((h/w)*size)), int(size)))
        else:
            img = resize(img, (int(size), int(round((w/h)*size))))
    return img

def image_scatter(features, images, img_res, perplexity, n_iter, res=4000, cval=1.):
    """
    Embeds images via tsne into a scatter plot.

    Parameters
    ---------
    features: numpy array
        Features to visualize

    images: list or numpy array
        Corresponding images to features. Expects float images from (0,1).

    img_res: float or int
        Resolution to embed images at

    res: float or int
        Size of embedding image in pixels

    cval: float or numpy array
        Background color value

    Returns
    ------
    canvas: numpy array
        Image of visualization
    """
    features = np.copy(features).astype('float64')
    # images = [gray_to_color(image) for image in images]
    images = [cv2.resize(image, img_res, interpolation = cv2.INTER_CUBIC) for image in images]
    # max_width = max([image.shape[0] for image in images])
    # max_height = max([image.shape[1] for image in images])
    max_width = img_res[0]
    max_height = img_res[1]

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)

    features = np.squeeze(features)
    f2d = tsne.fit_transform(features)

    xx = f2d[:, 0]
    yy = f2d[:, 1]
    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()
    # Fix the ratios
    sx = (x_max-x_min)
    sy = (y_max-y_min)
    if sx > sy:
        res_x = sx/float(sy)*res
        res_y = res
    else:
        res_x = res
        res_y = sy/float(sx)*res

    canvas = np.ones((res_x+max_width, res_y+max_height, 3))*cval
    x_coords = np.linspace(x_min, x_max, res_x)
    y_coords = np.linspace(y_min, y_max, res_y)
    for x, y, image in zip(xx, yy, images):
        w, h = image.shape[:2]
        x_idx = np.argmin((x - x_coords)**2)
        y_idx = np.argmin((y - y_coords)**2)
        canvas[x_idx:x_idx+w, y_idx:y_idx+h] = image
    return canvas


def main():
    model = sys.argv[1]
    year = sys.argv[2]
    month = sys.argv[3]
    perplexity = int(sys.argv[4])
    n_iter = int(sys.argv[5])

    start_time = time.time()

    preds_dir = join('/tigress/dchouren/thesis/', 'bottleneck3.h5')
    image_dir = join('/scratch/network/dchouren/images/', year, month, month)

    # preds = h5py.File(join(preds_dir, year + '_' + month + '.h5'))
    preds = h5py.File(preds_dir)

    embeddings = preds['bottlenecks'][:10000]
    filenames = preds['filenames'][:10000]

    images = []
    for im in filenames:
        try:
            x = cv2.imread(join(image_dir, im.decode('utf8')))
            images.append(x)
        except:
            pass

    images = np.array(images)
    # ipdb.set_trace()
    # images = np.array([cv2.imread(join(image_dir, im.decode('utf8'))) for im in filenames])
    
    canvas = image_scatter(embeddings, images, (64,64), perplexity, n_iter, res=8000)

    cv2.imwrite('/tigress/dchouren/thesis/plots/tsne/{}_{}_{}_{}.png'.format(year, month, perplexity, n_iter), canvas)

    print('{} | /tigress/dchouren/thesis/plots/tsne/{}_{}_{}_{}.png'.format(int(time.time() - start_time), year, month, perplexity, n_iter))



if __name__ == '__main__':
    sys.exit(main())

















