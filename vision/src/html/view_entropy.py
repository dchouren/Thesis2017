import sys
from html_utils import *

entropy_file = sys.argv[1]
output_file = sys.argv[2]

html_doc = begin_document()

with open(entropy_file, 'r') as inf:
    for line in inf:
        image_path, entropy = line.split(':')
        url = original_url_given_filepath(image_path)
        html_doc += add_image(url, 'Entropy:', entropy, '')
        # if float(entropy) > 5.0:


html_doc += end_document()
write_document(html_doc, output_file)
sys.exit(1)