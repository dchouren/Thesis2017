'''
python scrape_flickr2.py /tigress/dchouren/thesis/resources/paths 10000
'''

#!/usr/bin/python

#Image querying script written by Tamara Berg,
#and extended heavily James Hays

#9/26/2007 added dynamic timeslices to query more efficiently.
#8/18/2008 added new fields and set maximum time slice.
#8/19/2008 this is a much simpler function which gets ALL geotagged photos of
# sufficient accuracy.  No queries, no negative constraints.
# divides up the query results into multiple files
# 1/5/2009
# now uses date_taken instead of date_upload to get more diverse blocks of images
# 1/13/2009 - uses the original im2gps keywords, not as negative constraints though

import sys, string, math, time, socket
from flickrapi import FlickrAPI
from datetime import datetime
from os.path import join

import ipdb

socket.setdefaulttimeout(30)  #30 second time out on sockets before they throw
#an exception.  I've been having trouble with urllib.urlopen hanging in the
#flickr API.  This will show up as exceptions.IOError.

#the time out needs to be pretty long, it seems, because the flickr servers can be slow
#to respond to our big searches.

print(sys.argv)

if len(sys.argv) != 4:
  print(__doc__)
  sys.exit(0)

output_dir = sys.argv[1]
limit = int(sys.argv[2])
mintime = datetime.strptime(sys.argv[3], '%Y-%m-%d %H:%M:%S').timestamp()

print('Mintime: {}'.format(mintime))


API_KEY = '639ee716be3441f34a88df4f92747883'
SECRET = 'b0cb0ba28e8d7a34'

flickr = FlickrAPI(API_KEY, SECRET)

# query_file = open(query_file_name, 'r')

def search_flickr(page, time_start, time_end):
    f = flickr.photos_search(page=page, extras='geo, date_taken, views', media='photos', bbox='-74.052544, 40.525070, -73.740685, 40.889249', min_upload_date=time_start, max_upload_date=time_end)

    return f

infos = []
write_count = 0
query_count = 0
wrote_query = False

total_images_queried = 0

# number of seconds to skip per query
#timeskip = 62899200 #two years
timeskip = 604800  #one week
# timeskip = 172800  #two days
#timeskip = 86400 #one day
#timeskip = 3600 #one hour
#timeskip = 2257 #for resuming previous query

# mintime = 1325376000 # 1/1/2012
# mintime = 1332112633 # restart 11/5
# mintime = 1359400860 # restart 11/30
# mintime = 1374019200 # restart 12/23
# mintime = 1452729600 # restart 1/4
# mintime = 1454371200 # restart 1/5
# mintime = 1456099200
# mintime = 1462838400 
# mintime = 1466208000
# mintime = 1469404800
# mintime = 1474156800
# mintime = 1475366400
# mintime = 1477440000
# mintime = 1451606400 # restart 2/3, clean 2016 run
#Get the aliases and functions
maxtime = mintime+timeskip
endtime =  1478131200  # 11/03/2016

#this is the desired number of photos in each block
desired_photos = 250

print(datetime.fromtimestamp(mintime))
print(datetime.fromtimestamp(endtime))

while (maxtime < endtime):
    #new approach - adjust maxtime until we get the desired number of images
    #within a block. We'll need to keep upper bounds and lower
    #lower bound is well defined (mintime), but upper bound is not. We can't
    #search all the way from endtime.

    lower_bound = mintime + 900 #lower bound OF the upper time limit. must be at least 15 minutes or zero results
    upper_bound = mintime + timeskip * 20 #upper bound of the upper time limit
    maxtime     = .95 * lower_bound + .05 * upper_bound

    # print('\nBinary search on time range upper bound')
    # print('Lower bound is ' + str(datetime.fromtimestamp(lower_bound)))
    # print('Upper bound is ' + str(datetime.fromtimestamp(upper_bound)))

    keep_going = 6 #search stops after a fixed number of iterations
    while( keep_going > 0 and maxtime < endtime):

        try:
            rsp = search_flickr('1', mintime, maxtime)
            ##min_taken_date=str(datetime.fromtimestamp(mintime)),
            ##max_taken_date=str(datetime.fromtimestamp(maxtime)))
            #we want to catch these failures somehow and keep going.

            # ipdb.set_trace()
            time.sleep(1)
            # flickr.testFailure(rsp)
            total_images = rsp[0].get('total')
            null_test = int(total_images) #want to make sure this won't crash later on for some reason
            null_test = float(total_images)

            print('\nNum images: ' + total_images)
            print('Min time: ' + str(mintime) + ' Max time: ' + str(maxtime) + ' Timeskip:  ' + str(maxtime - mintime))

            if int(total_images) > desired_photos:
                # print('Too many photos in block, reducing maxtime')
                upper_bound = maxtime
                maxtime = (lower_bound + maxtime) / 2 #midpoint between current value and lower bound.

            if int(total_images) < desired_photos:
                # print('Too few photos in block, increasing maxtime')
                lower_bound = maxtime
                maxtime = (upper_bound + maxtime) / 2

            # print('Lower bound is ' + str(datetime.fromtimestamp(lower_bound)))
            # print('Upper bound is ' + str(datetime.fromtimestamp(upper_bound)))

            if int(total_images) > 0: #only if we're not in a degenerate case
                keep_going = keep_going - 1
            else:
                upper_bound = upper_bound + timeskip

        except KeyboardInterrupt:
            print('Keyboard exception while querying for images, exiting\n')
            raise
        except:
            print(sys.exc_info()[0])
            print ('Exception encountered while querying for images\n')

    #end of while binary search
    # print('Finished binary search')

    print('\nMintime: ' + str(datetime.fromtimestamp(mintime)) + ' Maxtime: ' + str(datetime.fromtimestamp(maxtime)) + ' Timeskip: ' + str(datetime.fromtimestamp(timeskip)))

    print('Num images: ' + total_images)

    current_image_num = 1

    num = int(rsp[0].get('pages'))
    print('Total pages: ' + str(num))

    #only visit 16 pages max, to try and avoid the dreaded duplicate bug
    #16 pages = 4000 images, should be duplicate safe.  Most interesting pictures will be taken.

    num_visit_pages = min(16,num)

    print('Visiting ' + str(num_visit_pages) + ' pages (up to ' + str(num_visit_pages * 250) + ' images)')

    total_images_queried += min((num_visit_pages * 250), int(total_images))

    #print 'stopping before page ' + str(int(math.ceil(num/3) + 1)) + '\n'

    for pagenum in range(0, num_visit_pages):
        f = search_flickr(pagenum, mintime, maxtime)

        for i, photo in enumerate(f[0]):
            try:
                id = photo.get('id')
                latitude = photo.get('latitude')
                longitude = photo.get('longitude')
                accuracy = photo.get('accuracy')
                owner = photo.get('owner')
                title = photo.get('title')
                views = photo.get('views')
                date_taken = photo.get('datetaken')
                description = flickr.photos.getInfo(photo_id=id)[0][2].get('description')
                sizes = flickr.photos.getSizes(photo_id=id)[0]
                medium = sizes[5]
                src_url = medium.get('source')
            except:
                continue

            info = ','.join([src_url, date_taken, latitude, longitude, accuracy, owner, title, str(description)])
            infos.append(info)

            query_count += 1

            if query_count >= limit:
                filename = join(output_dir, datetime.fromtimestamp(maxtime).strftime('%Y-%m-%d-%H-%M-%S'))
                with open(filename, 'w') as outf:
                    for info in infos:
                        outf.write(info + '\n')
                    print('Wrote {} to {}'.format(len(infos), str(filename)))
                wrote_query = True
                write_count += 1
                infos = []
                query_count = 0
                break

    timeskip = maxtime - mintime #used for initializing next binary search
    mintime  = maxtime


if not wrote_query:
    filename = join(output_dir, datetime.fromtimestamp(maxtime).strftime('%Y-%m-%d-%H-%M-%S'))
    with open(str(filename), 'w') as outf:
        for info in infos:
            outf.write(info + '\n')
        print('Wrote {} to {}'.format(len(infos), str(filename)))
    wrote_query = True
    write_count += 1
    infos = []
    query_count = 0














