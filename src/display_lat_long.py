'''
python src/display_lat_long.py resources/paths/2014_new/2014_all 40.706 -73.996 100
'''

import html.html_utils as html

import sys
import ipdb

if len(sys.argv) != 5:
  print(__doc__)
  sys.exit(0)


path_file = sys.argv[1]
lat = sys.argv[2]
lon = sys.argv[3]
n = sys.argv[4]

images = [line for line in open(path_file) if line.split(',')[2].startswith(lat) and line.split(',')[3].startswith(lon)]

# ipdb.set_trace()

doc = html.begin_document()
for image in images[:100]:
    tokens = image.split(',')
    args = tokens[:4] + [tokens[6]]
    doc += html.add_image(*args)

doc += html.end_document()
html.write_document(doc, 'test.html')
# print(images)
# 40.706031 -73.996892