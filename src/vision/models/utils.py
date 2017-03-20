import ipdb
import sys
from os.path import join
import h5py

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Dense, Activation, Flatten


weights_dir = '/tigress/dchouren/thesis/resources/weights/'

def _load_model(model_name, include_top=True, weights='imagenet'):
    if model_name == 'resnet50':
        from models.resnet50 import ResNet50
        model = ResNet50(weights=weights, include_top=include_top)
    elif model_name == 'vgg16':
        from models.vgg16 import VGG16
        model = VGG16(weights=weights, include_top=include_top)
        if weights == 'hybrid1365':
            model.load_weights(join(weights_dir, 'vgg16_hybrid1365_weights.h5'))
        elif weights == 'places365':
            model.load_weights(join(weights_dir, 'vgg16_places365_weights.h5'))
    elif model_name == 'vgg19':
        from models.vgg19 import VGG19
        model = VGG19(weights=weights, include_top=include_top)
    elif model_name == 'inception':
        from models.inception_v3 import InceptionV3
        model = InceptionV3(weights=weights, include_top=include_top)
    elif model_name == 'vgg16_hybrid1365':
        from models.vgg16 import VGG16
        model = VGG16(weights=weights, include_top=include_top)
        model.load_weights(join(weights_dir, 'vgg16_hybrid1365_weights.h5'), by_name=True)
    elif model_name == 'vgg16_places365':
        from models.vgg16 import VGG16
        model = VGG16(weights=weights, include_top=include_top)
        model.load_weights(join(weights_dir, 'vgg16_places365_weights.h5'), by_name=True)
    elif model_name == 'resnet152_hybrid1365':
        from models.resnet import ResnetBuilder
        model = ResnetBuilder.build_resnet_152((3, 224, 224), 100)
        print(join(weights_dir, 'resnet152_hybrid1365.h5'))
        model.load_weights(join(weights_dir, 'resnet152_hybrid1365.h5'), by_name=True)
    else:
        print ('Not a valid model. Valid models are vgg16, vgg19, resnet50, inception,  vgg16_hybrid1365, and vgg16_places365')
        sys.exit(1)

    return model


def _load_sequential_model(model_name, weights='imagenet'):
    if model_name == 'vgg16':
        img_width, img_height = 224,224
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        f = h5py.File(join('/spindle/', 'resources/weights/imagenet/{}_weights.h5'.format(model_name)))
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()

        return model

    elif model_name == 'resnet50':
        x = ZeroPadding2D((3, 3))(img_input)
        x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = AveragePooling2D((7, 7), name='avg_pool')(x)

