import sys
from os.path import join
import io
import string

import ipdb

output_dir = sys.argv[1]
batch_size = int(sys.argv[2])

def photo_url_given_path( path ):
    if type(path) == str:
        return "https://media-cdn.tripadvisor.com/media/" + path.replace('photo-o','photo-s')
    return "https://media-cdn.tripadvisor.com/media/" + path.str.replace('^photo-o','photo-s')


def write_fpaths(batch_to_write, output_file):
    with open(output_file, 'w') as outf:
        for line in batch_to_write:
            try:
                outf.write(line)
            except:
                print('Error writing: {}'.format(line))


if __name__ == "__main__":
    # input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

    same_location = False
    prev_locationid = -1
    count = 0
    batch_to_write = []
    write_count = 0
    for path in sys.stdin:
        photo_suffix, mediaid, locationid = path.split('\t')
        if photo_suffix.split('-')[0] != 'photo':
            continue

        if locationid == prev_locationid:
            same_location = True
        else:
            same_location = False

        prev_locationid = locationid

        if count >= batch_size and not same_location:
            output_file = join(output_dir, str(write_count))
            write_fpaths(batch_to_write, output_file)
            write_count += 1
            print('Wrote {} with {} lines'.format(output_file, len(batch_to_write)))
            count = 0
            batch_to_write = []

        fpath = photo_url_given_path(photo_suffix)
        full_line = ','.join([fpath, mediaid, locationid])
        batch_to_write.append(full_line)
        count += 1

import sys
import io
import string

import ipdb

output_file = sys.argv[1]
limit = int(sys.argv[2])

def photo_url_given_path( path ):
    if type(path) == str:
        return "https://media-cdn.tripadvisor.com/media/" + path.replace('photo-o','photo-s')
    return "https://media-cdn.tripadvisor.com/media/" + path.str.replace('^photo-o','photo-s')


if __name__ == "__main__":
    # input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

    same_location = False
    prev_locationid = -1
    count = 0
    batch_to_write = []
    for path in sys.stdin:
        photo_suffix, mediaid, locationid = path.split('\t')
        if photo_suffix.split('-')[0] != 'photo':
            continue

        if locationid == prev_locationid:
            same_location = True
        else:

        if count >= limit and not same_location:
            write_fpaths(batch_to_write)


        fpath = photo_url_given_path(photo_suffix)
        full_line = ','.join([fpath, mediaid, locationid])
        batch_to_write.append(full_line)
        count += 1




    with open(output_file, 'w') as outf:
        for path in sys.stdin:

            try:
                fpath = photo_url_given_path(photo_suffix)
                full_line = ','.join([fpath, mediaid, locationid])
                outf.write(full_line)
            except:
                print ("error: " + path)