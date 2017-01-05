'''
Convert hdf5 object to menu training object
Usage: python pull_menu_images.py [f5_file] [training_file]
Example: python src/pull_menu_images.py assets/f5/yelpset.f5 assets/f5/yelpset.training
'''

import sys
import h5py
import numpy as np
from itertools import compress

import ipdb

def main():

    with h5py.File(hdf5_file, 'r') as inf:
        input = inf.get('input')
        mapping = inf.get('mapping')
        output = inf.get('output')

        input = list(compress(input, [output[x] == 1.0 for x in output]))
        output = np.asarray([922 for x in output if x == 5], dtype=np.float64)

        with h5py.File(training_file, 'w') as outf:
            d1 = outf.create_dataset('input', data=input)
            d2 = outf.create_dataset('output', data=output)

            outf.close()
            inf.close()




if __name__ == "__main__":
    if len(sys.argv) != 3:
        print (__doc__)
        sys.exit(0)

    hdf5_file = sys.argv[1]
    training_file = sys.argv[2]

    main()