import numpy as np
import json

from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras import backend as K

import ipdb

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
hybrid_map_file = 'resources/json/hybrid_class_index.json'


def mean_center(x, data_format='channels_first'):
    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        # x = x[:, ::-1, :, :]
        # Zero-center by mean pixel
        x[:, 0, :, :] -= 94.78833771
        x[:, 1, :, :] -= 91.02941132
        x[:, 2, :, :] -= 88.50362396
    else:
        # 'RGB'->'BGR'
        # x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, :, 0] -= 94.78833771
        x[:, :, :, 1] -= 91.02941132
        x[:, :, :, 2] -= 88.50362396

    return x


def load_and_preprocess_image(im_path, dataset='imagenet', x_size=224, y_size=224, preprocess=True, rescale=False):
    try:
        img = image.load_img(im_path, target_size=(x_size, y_size))
    except:
        print('failure opening: {}'.format(im_path))
        return None
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # if data_format is None:
    data_format = K.image_data_format()
    if preprocess:
        # x = preprocess_input(x, dataset=dataset)
        x = mean_center(x, data_format)
    if rescale:
        x *= 1./255
    return x


def preprocess_input(x, dataset='imagenet', dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    # default imagenet values
    r_val = 103.939
    g_val = 116.779
    b_val = 123.680

    if dataset == 'hybrid':
        r_val = 103.955
        g_val = 116.415
        b_val = 123.088
    elif dataset == 'places365':
        r_val = 104.113
        g_val = 112.812
        b_val = 117.230
    elif dataset == 'flickr':
        r_val = 94.78833771
        g_val = 91.02941132
        b_val = 88.50362396

    if dim_ordering == 'th':
        x[:, 0, :, :] -= r_val
        x[:, 1, :, :] -= g_val
        x[:, 2, :, :] -= b_val
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= r_val
        x[:, :, :, 1] -= g_val
        x[:, :, :, 2] -= b_val
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x

#####
## FOR IMAGENET ONLY
# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


class_indexes = {
    'imagenet': 'imagenet_class_index.json',
    'hybrid': 'hybrid_class_index.json',
}

def decode_predictions(preds):
    global CLASS_INDEX
    assert len(preds.shape) == 2 and preds.shape[1] == 1000
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    indices = np.argmax(preds, axis=-1)
    results = []
    for i in indices:
        results.append(CLASS_INDEX[str(i)])
    return results

all_maps = {
    'menu': {0: 'menu', 1: 'other', 2: 'receipt'},   # original menu classifier
    'thumbnail': {0: 'bad_thumbnail', 1: 'good_thumbnail'},
    'dishes': {0: 'pasta', 1:'seafood', 2:'steak', 3:'sushi'},
    'thumbnail3': {0: 'bad', 1: 'good_dining', 2: 'good_food'},   # flickr model
    'gfbf': {0: 'good_food', 1: 'bad_food', 2: 'non_food'},   # flickr model
    'gfbfgd': {0: 'good_dining', 1:'good_food', 2:'bad_food', 3: 'non_food'},   # flickr model
    'dslr': {0: 'dslr', 1: 'nondslr'},   # dslr
}

def _decode_predictions(all_predictions, mapping):

    if mapping == 'imagenet':
        return decode_predictions(all_predictions)

    decoded_predictions = []
    preds_map = all_maps[mapping]

    for predictions in all_predictions:
        max_index = np.argmax(predictions)
        prob = predictions[max_index]

        decoded_predictions.append([preds_map[max_index], prob])

    return decoded_predictions



# def standardize(x):
    # if self.rescale:
    #     x *= self.rescale
    # x is a single image, so it doesn't have image number at index 0
    # img_channel_index = self.channel_index - 1
    # if self.samplewise_center:
    #     x -= np.mean(x, axis=img_channel_index, keepdims=True)
    # if self.samplewise_std_normalization:
    #     x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)
    #
    # if self.featurewise_center:
    #     x -= self.mean
    # if self.featurewise_std_normalization:
    #     x /= (self.std + 1e-7)
    #
    # if self.zca_whitening:
    #     flatx = np.reshape(x, (x.size))
    #     whitex = np.dot(flatx, self.principal_components)
    #     x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
    #
    # return x
