'''
usage: python examine_output.py [geo] [file_grep_pattern] [limit]
example: python src/examine_output.py 187147 '*' 1000
'''

import sys
from os.path import join
import string
import glob
import re

import ipdb



def clean_string(s):
    return re.sub('[^\s!-~]', '', s)


def gen_html_image_str(locationid, pred_line):
    mediaid, url, prediction, p1, p2, p3, p4, p5 = pred_line
    p1 = float(p1)
    p2 = float(p2)
    p3 = float(p3)
    p4 = float(p4)
    p5 = float(p5)
    html_string = '<h4>' + prediction + ': ' + '{0:.2f}'.format(max([p1,p2,p3,p4,p5])) + '</h4>\n'
    html_string += '<p>MediaID: ' + mediaid + '</p>'
    html_string += '<img src=\"' + url + '\" alt=\"' + prediction + '\">\n'
    html_string += '<br><br>\n\n'
    return html_string
# <img src="pic_mountain.jpg" alt="Mountain View" style="width:304px;height:228px;">

def write_html_page(html_doc, output_page):
    html_doc += '''</body>
                </html>'''

    print('Outputted to: ' + output_page)

    with open(output_page, 'w') as outf:
        outf.write(html_doc)


def main():
    total_read = 0

    count = 0

    location_info_map = {}

    for pred_file in pred_files:
        print (pred_file)

        with open(pred_file, 'r') as inf:
            next(inf)
            for pred_line in inf.readlines():
                pred_line = clean_string(pred_line).strip()
                locationid, mediaid, image_url, prediction, p1,p2,p3,p4,p5 = pred_line.split(',')
                mediaid = mediaid.strip('n')

                if not prediction == 'menu':
                    continue

                try:
                    # ipdb.set_trace()
                    location_info_map[locationid].append([mediaid, image_url, prediction, p1,p2,p3,p4,p5])
                except KeyError:
                    location_info_map[locationid] = [[mediaid, image_url, prediction, p1,p2,p3,p4,p5]]

    html_doc = '''
                <!DOCTYPE html>
                <html>
                <body>
                '''

    for locationid, info_group in sorted(location_info_map.items(), key=lambda x: int(x[0])):

        if total_read + len(info_group) > limit:
            write_html_page(html_doc, join(html_base, 'all_' + str(count) + '.html'))
            count += 1
            html_doc = '''
                    <!DOCTYPE html>
                    <html>
                    <body>
                    '''
            total_read = 0

        total_read += len(info_group)

        locationid = locationid.strip('n')
        html_doc += '<hr>'
        html_doc += '<h2>' + locationid + '</h2>'
        html_doc += '<a href=\"http://www.tripadvisor.com/' + locationid + '\">the_restaurant</a><br>'
        info_group = sorted(info_group, key=lambda x: x[4], reverse=True)
        for info in info_group:
            html_image_string = gen_html_image_str(locationid, info)
            html_doc += html_image_string


    write_html_page(html_doc, join(html_base, 'all_' + str(count) + '.html'))
    sys.exit()




if __name__ == "__main__":
    if len(sys.argv) != 4:
        print (__doc__)
        sys.exit(0)


    geo = sys.argv[1]
    file_grep = sys.argv[2]
    pred_dir = join('output', geo)
    pred_files = glob.glob(join(pred_dir, file_grep))
    html_base = join('html', geo)
    limit = int(sys.argv[3])

    main()