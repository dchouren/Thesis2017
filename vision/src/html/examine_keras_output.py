'''
usage: python examine_output.py [task] [dataset] [file_grep_pattern] [desired_label] [probability_cutoff] [limit]
example: python src/examine_keras_output.py thumbnail flickr 'thumbnail3_df' good 0.8 1000
python src/html/examine_keras_output.py thumbnail dslr 'dslr' dslr 0 1000

python src/html/examine_keras_output.py thumbnail dslr 'finetuned_menu_face_dslr_milan' good 0.5 1000

Create html files from the output files in output/[task]/[dataset]/[file_grep_pattern] where the prediction contains [desired_label] and with a probability >= [probability_cutoff]. Writes [limit] images per html file.
'''

import sys
from os.path import join
import string
import glob
import re
import os

from html_utils import *

import ipdb

def load_thumbnails(thumbnail_file):
    location_thumbnail_map = {}
    with open(thumbnail_file, 'r') as inf:
        for line in inf:
            locationid, thumbnail_url = line.split(',')
            location_thumbnail_map[locationid] = thumbnail_url.replace('{','').replace('}','')
    return location_thumbnail_map


def group_by_location(pred_files):

    location_info_map = {}
    seen_locations = []

    pred_files = sorted(pred_files)

    for pred_file in pred_files:
        print (pred_file)

        with open(pred_file, 'r') as inf:
            try:
                next(inf)
            except:
                continue
            for pred_line in inf.readlines():
                pred_line = clean_string(pred_line).strip()
                tokens = pred_line.split(',')

                prediction = tokens[-2]
                probability = float(tokens[-1])
                locationid = tokens[2]

                if not locationid in seen_locations:
                    seen_locations.append(locationid)

                if not label == prediction and label != 'all':
                    continue
                if probability < probability_cutoff:
                    continue

                try:
                    # ipdb.set_trace()
                    location_info_map[locationid].append(tokens)
                except KeyError:
                    location_info_map[locationid] = [tokens]

    return location_info_map, seen_locations


def main():
    total_read = 0
    total_images = 0

    count = 0
    location_info_map, seen_locations = group_by_location(input_files)

    # thumbnail_file = 'resources/187147_thumbnails'
    thumbnail_file = 'resources/thumbnails/milan_thumbnails'
    location_thumbnail_map = load_thumbnails(thumbnail_file)

    html_doc = begin_document()

    total_owner = 0
    locs_w_owner = 0
    # for location, infos in location_info_map.items():
    #     location_has_info = False
        # for info in infos:
        #     # if info[5] == 'true':
        #     #     total_owner += 1
        #     #     location_has_info = True
        # if location_has_info:
        #     locs_w_owner += 1


    # ipdb.set_trace()
    num_not_blur = 0
    good_images = 0
    for locationid, info_groups in sorted(location_info_map.items(), key=lambda x: float(x[0])):
        has_not_blur = False
        total_images += len(info_groups)

        if total_read + len(info_groups) > limit:
            html_filename = gen_html_filename(output_base, label, grep, probability_cutoff, count)
            html_doc += end_document()
            write_document(html_doc, html_filename)
            count += 1
            html_doc = begin_document()
            total_read = 0

        total_read += len(info_groups)

        try:
            thumbnail_url = location_thumbnail_map[locationid]
        except:
            thumbnail_url = ''
        html_doc += add_location(locationid, thumbnail_url)

        # ipdb.set_trace()
        other = ''
        for info in sorted(sorted(info_groups, key=lambda x: float(x[-1]), reverse=True), key=lambda x: x[-2], reverse=True):
            url = info[0]
            probability = info[-1]
            prediction = info[-2]

            # other = '{0:.6f}'.format(float(info[-3]))
            # if float(info[-3]) > 2.0:
            #     has_not_blur = True
            # else:
            #     continue

            good_images += 1
            html_doc += add_image(url, prediction, probability, other)

        if has_not_blur:
            num_not_blur += 1

        html_doc += end_location()

    print('Total locations with {} images: {}'.format(label, len(location_info_map.keys())))
    print('Total locations: {}'.format(len(seen_locations)))
    print('Total location with not blur: {}'.format(num_not_blur))
    print('Total {} images: {}'.format(label, good_images))


    html_filename = gen_html_filename(output_base, label, grep, probability_cutoff, count)
    html_doc += end_document()
    write_document(html_doc, html_filename)
    sys.exit()





if __name__ == "__main__":
    if len(sys.argv) != 7:
        print (__doc__)
        sys.exit(0)


    task = sys.argv[1]
    dataset = sys.argv[2]
    grep = sys.argv[3]
    input_files = glob.glob(join('output', task, dataset, grep))
    print(input_files)
    label = sys.argv[4]
    probability_cutoff = float(sys.argv[5])
    limit = int(sys.argv[6])
    output_base = join('html', task, dataset)

    if not os.path.exists(output_base):
        os.makedirs(output_base)

    main()