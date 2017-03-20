import sys
from os.path import join

year = sys.argv[1]
fpath_base_dir = '/tigress/dchouren/thesis/resources/paths/' + year + '_new'

fpath_file = join(fpath_base_dir, year + '_all')
with open(fpath_file, 'r') as inf:
    fpaths = inf.readlines()

split_fpaths = {}
for line in fpaths:
    tokens = line.split(',')
    date = tokens[1]
    tyear, month = date.split('-')[0], date.split('-')[1]
    if not tyear == year:
        continue
    rearranged_tokens = [tokens[2], tokens[3], tokens[0], tokens[1], *tokens[4:]]

    if month in split_fpaths:
        split_fpaths[month] += [rearranged_tokens]
    else:
        split_fpaths[month] = [rearranged_tokens]


for month, fpaths in split_fpaths.items():
    outf = open(join(fpath_base_dir, month), 'w')

    for line in fpaths:
        outf.write(','.join(line))
