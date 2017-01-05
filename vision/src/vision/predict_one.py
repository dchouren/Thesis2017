'''
Used for quick testing
'''

import glob
import ipdb
import numpy as np
import sys
from keras.models import load_model
from keras.preprocessing import image
import h5py
from os.path import join


from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

from models.utils import _load_model
from vision_utils import load_and_preprocess_image, _decode_predictions


if len(sys.argv) != 4:
    print (__doc__)
    sys.exit(0)


def get_filename(filepath):
    return filepath.split('/')[-1]

x_size, y_size = 224, 224

# model_name = sys.argv[1]
# second_model = sys.argv[2]
# im_path = sys.argv[3]
im_dir = sys.argv[1]

base_model = _load_model('vgg16', include_top=False)

ipdb.set_trace()

model = load_model('/spindle/dchouren/resources/thumbnail/dslr/trained_models/top_resnet50_100_55707_dslr.h5')

# menu_model = load_model('old_menu.h5')
# # menu_model = load_model('/spindle/dchouren/resources/textlike/menu_receipt/trained_models/top_vgg16_30_31089_text.h5')
#
# model = Sequential()
# model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
#
# model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# #model = _load_model(model_name, include_top=True)
# #model.load_weights('/spindle/dchouren/resources/weights/imagenet/vgg16_weights.h5', by_name=True)
#
# f = h5py.File('/spindle/dchouren/resources/weights/imagenet/{}_weights.h5'.format('vgg16'))
# for k in range(f.attrs['nb_layers']):
#     if k >= len(model.layers):
#         # we don't look at the last (fully-connected) layers in the savefile
#         break
#     g = f['layer_{}'.format(k)]
#     weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
#     model.layers[k].set_weights(weights)
# f.close()
#
# model.add(menu_model)
# model.save('/spindle/dchouren/resources/menu/187147/trained_models/old_menu.h5')

# ipdb.set_trace()
image_paths = glob.glob(join(im_dir, '*.jpg'))

# print(image_paths)
all_preds = []
# get_layer_output = K.function([model.layers[0].input], [model.layers[20].output])

for im_path in image_paths:

    x = load_and_preprocess_image(im_path, x_size, y_size, True)
    # base_preds = base_model.predict(x)

    preds = base_model.predict(x)[0]
    # pred, prob = _decode_predictions(preds, 'dslr')

    ipdb.set_trace()
    print(im_path, pred, prob)




# ipdb.set_trace()