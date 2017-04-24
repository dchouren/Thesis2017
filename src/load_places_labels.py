import numpy as np
import glob
import h5py

from keras.preprocessing import image


import ipdb


labels_file = '/tigress/dchouren/thesis/resources/places365/images/places365_val.txt'

labels = []
with open(labels_file, 'r') as inf:
    for line in inf.readlines():
        one_hot = [0]*365
        label = int(line.split(' ')[-1])
        one_hot[label] = 1
        # labels += [line.split(' ')[-1]]
        labels += [one_hot]

# labels = [label.encode('utf8') for label in labels]

labels = np.array(labels)
# np.save('/tigress/dchouren/thesis/resources/places365/images/val_labels.npy', labels)



def load_and_preprocess_image(im_path, x_size=224, y_size=224, rescale=False):
    try:
        img = image.load_img(im_path, target_size=(x_size, y_size))
    except:
        print('failure opening: {}'.format(im_path))
        return None
    x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    if rescale:
        x *= 1./255
    return x


filenames = [img for img in glob.glob("/tigress/dchouren/thesis/resources/places365/images/val/*.jpg")]

filenames.sort() # ADD THIS LINE

images_file = '/tigress/dchouren/thesis/resources/places365/images/val.h5'
batch_size = 3650
with h5py.File(images_file, 'w') as outf:
    images = []
    dset = outf.create_dataset('images', shape=(36500,3,224,224))
    row_count = 0
    for img in filenames:
        x = load_and_preprocess_image(img, rescale=True)
        images.append(x)

        if len(images) == batch_size:
            print('batch')
            dset[row_count:row_count+batch_size] = np.asarray(images)
            images = []
            row_count += batch_size

    labels_dset = outf.create_dataset('labels', data=labels)




    # np.save('/tigress/dchouren/thesis/resources/places365/images/val_images.npy', images)

ipdb.set_trace()

