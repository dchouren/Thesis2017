from keras.preprocessing import image
import subprocess as sub


if __name__ == '__main__':

    import sys
    import glob
    from os.path import join

    im_dir = sys.argv[1]
    images = glob.glob(join(im_dir, '/*/*.jpg'))

    for image_path in images:
        try:
            im = image.load_img(image_path)
        except:
            print('removed {}'.format(image_path))
            # sub.call(["rm", "-f", image_path])

