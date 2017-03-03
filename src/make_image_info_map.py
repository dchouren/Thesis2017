import sys
import glob
from os.path import join
import json


paths_dir = sys.argv[1]


for year in glob.glob(join(paths_dir, '*')):

    image_info_map = {}
    for path_file in glob.glob(join(paths_dir, year, '*')):
        with open(path_file, 'r') as inf:
            for line in inf:
                url = line.split(',')[0]
                url_id = url.split('/')[-1].split('.')[0]
                image_info_map[url_id] = line.split(',')

    with open(join('/tigress/dchouren/thesis/resources/maps/', year + '.json'), 'w') as outf:
        json.dump(image_info_map, outf)




if len(sys.argv) != 4:
  print(__doc__)
  sys.exit(0)