import glob
import h5py


filenames = glob.glob('/tigress/dchouren/thesis/resources/pairs/*.h5')

for filename in filenames:
    try:

        f = h5py.File(filename)
        x = f['pairs']
        print('{}: {}'.format(filename, x.shape))

        f.close()
    except:
        print(filename)

