'''
Utils used for creating html documents
'''

import re
from os.path import join
import glob


def clean_string(s):
    return re.sub('[^\s!-~]', '', s)


def original_url_given_filepath(filepath):
    image_name = filepath.split('/')[-1]
    image_name = '/'.join(image_name.split('-')[:4]) + '/' + '-'.join(image_name.split('-')[4:])
    url = 'https://media-cdn.tripadvisor.com/media/photo-s/' + image_name
    return url


def photo_url_given_path( path ):
    if type(path) == str:
        return "https://media-cdn.tripadvisor.com/media/" + path.replace('photo-o','photo-s')
    return "https://media-cdn.tripadvisor.com/media/" + path.str.replace('^photo-o','photo-s')


def original_url_given_path(path):
    return path.replace('photo-s','photo-o')


def begin_document():
    doc = "<html><body style='font-family:sans-serif'>\n"
    return doc


def add_location(locationid, thumbnail_url=None, second_thumbnail=None):
    s = "</div></div><div style='font-family:sans-serif;border-bottom:1px solid " \
        "gray;padding:5px;margin-bottom:30px;padding-bottom:30px'>\n \
    <div style='font-size:16px;color:#555;margin-bottom:10px;'>Photos for location: <b><a href='http://www.tripadvisor.com/" + locationid + "'>" + locationid + "</a></b></div>\n"

    if thumbnail_url:
        s += "<div style='font-size:16px;color:#555;margin-bottom:10px;'>Current thumbnail <img src='" + thumbnail_url + "' style='max-height:140px;></div><br>\n"

    if second_thumbnail:
        s += "<div style='font-size:16px;color:#555;margin-bottom:10px;'>Ditto: <img src='" + second_thumbnail + "' style='max-height:140px;></div><br>\n"

    s += "<div style='display:block'>"
    return s


def end_location():
    s = "</div>\n"
    return s


def add_image(url, prediction, probability, other):
    try:
        probability = '{0:.2f}'.format(float(probability))
    except:
        probability = probability
    s = "<div style='display:inline-block;margin:4px;text-align:center;'>\n<img src='" + url.replace('{','').replace(
        '}','') \
        + \
        "' style='max-height:140px;'><br/>\n<span style='font-size:12px;width:100%;'>" + prediction + " " + \
         probability + "<br>\n" + other + "</span>\n</div>\n"
    return s


def end_document():
    s = "</div></div>\n</body></html>\n"
    return s


def write_document(html_doc, output_page):
    with open(output_page, 'w') as outf:
        outf.write(html_doc)
    print('Outputted to: ' + output_page)


def gen_html_filename(html_base, label, grep, probability_cutoff, count):
    grep = grep.replace('*','')
    return join(html_base, grep + '_' + label + '_' + str(probability_cutoff) + '_' + str(count) + '.html')


def load_thumbnails(thumbnail_file):
    location_thumbnail_map = {}
    with open(thumbnail_file, 'r') as inf:
        for line in inf:
            locationid, thumbnail_url = line.split(',')
            location_thumbnail_map[locationid] = thumbnail_url.replace('{','').replace('}','')
    return location_thumbnail_map


def group_by_location(pred_files, label, probability_cutoff):
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

                if not label in prediction and label != 'all':
                    continue
                if probability < probability_cutoff:
                    continue

                try:
                    # ipdb.set_trace()
                    location_info_map[locationid].append(tokens)
                except KeyError:
                    location_info_map[locationid] = [tokens]
    return location_info_map, seen_locations


def pull_ditto_best_image(ditto_dir):
    html_files = glob.glob(join(ditto_dir, '*.html'))

    location_url_map = {}

    for html_file in html_files:
        with open(html_file, 'r') as inf:
            lines = '\n'.join(inf.readlines())
            location_tokens = lines.split('Photos for location')[1:]  # first token is not a location
            for location_token in location_tokens:
                locationid = location_token.split('</b>')[0].split(' : ')[-1]
                first_image = location_token.split('\' style')[0].split('src=\'')[-1]
                probability = location_token.split(' </span>')[0].split(';\'>')[-1]

                location_url_map[locationid] = (first_image, probability)

    return location_url_map