'''
Convert five class hdf5 files to binary file
Usage: python convert_to_binary_classification.py [f5_file] [f2_file]
Example: python src/convert_to_binary_classification.py assets/f5/yelpset.f5 assets/f1/yelpset.f1
'''

import sys
import h5py
import numpy as np

import ipdb

def main():

    with h5py.File(hdf5_file, 'r') as f5_obj:
        input = f5_obj.get('input')
        mapping = f5_obj.get('mapping')
        output = f5_obj.get('output')
        output = np.asarray([1.0 if x != 5 else 2.0 for x in output], dtype=np.float64)
        # f5_obj.close()

        with h5py.File(binary_file, 'w') as bin_obj:
            d1 = bin_obj.create_dataset('input', data=input)
            d2 = bin_obj.create_dataset('mapping', data=mapping)
            d3 = bin_obj.create_dataset('output', data=output)

            bin_obj.close()
            f5_obj.close()




if __name__ == "__main__":
    if len(sys.argv) != 3:
        print (__doc__)
        sys.exit(0)

    hdf5_file = sys.argv[1]
    binary_file = sys.argv[2]

    main()