'''
Convert extract images from an hdf5 file with an input, output, and mapping datasets containing the images, labels, and their urls

Usage: python examine_hdf5.py [hdf5_file] [output_dir]
Example: python src/examine_hdf5.py assets/f5/yelpset.f5 resources/f5_images

'''
import sys
import h5py
import scipy.misc
from os.path import join
import os
import numpy as np
import urllib.request
import glob


import ipdb


def write_html_page(html_doc, output_page):
    html_doc += '''</body>
                </html>'''

    with open(output_page, 'w') as outf:
        outf.write(html_doc)


def main():

    h1 = h5py.File(hdf5_file, 'r')
    images = h1.get('input')
    output = h1.get('output')
    mapping = h1.get('mapping')

    all_images = glob.glob(join(output_dir, '*'))
    for f in all_images:
        os.remove(f)

    output_map = {1: 'food', 2: 'drink', 3: 'inside', 4: 'outside', 5: 'menu'}

    urls = []

    image_count = 0
    for i, (image, y, url) in enumerate(zip(images, output, mapping)):
        if y != 5:
            continue
        image_count += 1
        # if image_count > 1000:
        #     break
        # image = np.swapaxes(image, 0, 1)
        # image = np.swapaxes(image, 1, 2)
        # scipy.misc.imsave(join(output_dir, 'image_' + str(i) + '.jpg'), image)

        url = url[0].decode('utf-8')
        urls.append(url)
        try:
            urllib.request.urlretrieve(url, join(output_dir, 'image_' + str(i) + '.jpg'))
        except:
            continue
        # urls.append(url[0].decode('utf-8'))
}}

    html_doc = '''
        <!DOCTYPE html>
        <html>
        <body>
        '''
    for i, url in enumerate(urls):
        html_string = '<h4> ' + str(i) + '</h4>'
        html_string += '<img style="max-height: 400px" src=\"' + url + '\" alt=\"nothing\">\n'
        html_string += '<br><br>\n\n'
        html_doc += html_string

    write_html_page(html_doc, join(output_dir, 'images.html'))



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print (__doc__)
        sys.exit(0)

    hdf5_file = sys.argv[1]
    output_dir = sys.argv[2]

    main()