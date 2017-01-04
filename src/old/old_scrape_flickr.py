from flickrapi import FlickrAPI, shorturl
import urllib.request
import os
import sys
import datetime
from os.path import join

import ipdb


API_KEY = '2b601611576cb460560f1c11cf6cbb45'
SECRET = '654a32a8f9ce451d'

PAGE_SIZE = 250

flickr = FlickrAPI(API_KEY, SECRET)

if len(sys.argv)>1:
    limit = int(sys.argv[1])
else:
    print ('no limit specified')

# queries = ['sushi']

# timeskip = 604800  #one week
# timeskip = 172800  #two days
timeskip = 86400 #one day
jan_1_2016 = 1451606400
jan_1_2006 = 1136073600
oct_09_2008 = 1223510400

time_start = oct_09_2008
time_end = time_start + timeskip

output_dir = '/tigress/dchouren/thesis/resources/paths'

while time_end < jan_1_2016:
    print(time_start)

    f = flickr.photos_search(bbox='-74.052544, 40.525070, -73.740685, 40.889249', min_upload_date=time_start, max_upload_date=time_end)
    num_pages = int(f[0].get('pages'))
    print(num_pages)
    infos = [] #store a list of what was downloaded

    # ipdb.set_trace()

    write_count = 0

    for page in range(0, num_pages):
        # print(i)
        f = flickr.photos_search(page=page, extras='geo', media='photos', bbox='-74.052544, 40.525070, -73.740685, 40.889249', min_upload_date=time_start, max_upload_date=time_end)
        for i, photo in enumerate(f[0]):
            id = photo.get('id')
            latitude = photo.get('latitude')
            longitude = photo.get('longitude')
            accuracy = photo.get('accuracy')
            owner = photo.get('owner')
            title = photo.get('title')
            try:
                description = flickr.photos.getInfo(photo_id=id)[0][2].get('description')
                sizes = flickr.photos.getSizes(photo_id=id)[0]
                medium = sizes[5]
                src_url = medium.get('source')
            except:
                continue

            info = ','.join([src_url, latitude, longitude, accuracy, owner, title, str(description)])
            infos.append(info)

        if page * PAGE_SIZE >= limit:
            filename = datetime.datetime.fromtimestamp(time_start).strftime('%Y-%m-%d_') + str(write_count)
            write_count += 1
            with open(filename, 'w') as outf:
                for info in infos:
                    outf.write(info + '\n')
                print('Wrote {} to {}'.format(len(infos), filename))
            infos = []
            # break
    filename = datetime.datetime.fromtimestamp(time_start).strftime('%Y-%m-%d_') + str(write_count)
    write_count += 1
    with open(join(output_dir, filename), 'w') as outf:
        for info in infos:
            outf.write(info + '\n')
        print('Wrote {} to {}'.format(len(infos), filename))
    time_start = time_end
    time_end += timeskip






