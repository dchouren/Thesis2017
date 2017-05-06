import sys
from os.path import join

import numpy as np

import html.html_utils as hm



preds = None
if len(sys.argv) > 1:
    preds_file = sys.argv[1]

    preds_file = join('/tigress/dchouren/thesis/evaluation/preds', preds_file)

    preds = np.load(preds_file)

doc = hm.begin_document()

base_preds = np.load(join('/tigress/dchouren/thesis/evaluation/preds', 'base_preds.npy'))

with open('/tigress/dchouren/thesis/evaluation/query_and_triplets.txt', 'r') as inf:

    for i, line in enumerate(inf.readlines()):
        index = i % 4
        line = line.strip()
        if index == 0:
            label = line

        if index == 1:
            doc += hm.open_div()
            if not preds is None:
                doc += hm.add_image(line, str(preds[int(i/4)]) + ' ' + str(base_preds[int(i/4)]))
            else:
                doc += hm.add_image(line, label)
        elif index == 2:
            doc += hm.add_image(line, label + '_pos')
        elif index == 3:
            doc += hm.add_image(line, label + '_neg')
            doc += hm.close_div()

doc += hm.end_document()

hm.write_document(doc, '/tigress/dchouren/thesis/evaluation/triplets.html')


