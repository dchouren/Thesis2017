"""
Scrape photos from Flickr API for a given search.
Usage: python scrape_flickr2.py [search_type] [query] [output_base_dir] [total_images_wanted] [limit]
Example: python src/scrape_flickr2.py group 34542434@N00 resources/paths/thumbnail/flickr 10000 1000
Example 2: python src/scrape_flickr2.py search dining\ restaurant resources/paths/thumbnail/flickr 10000 1000

[search type]: search_flickr function has option to search for photos by group_id or by a simple text search
[query]: the desired group_id or search text
[output_base_dir]: base dir where scraped urls should be written. output will create a subdir for the query name
[total_images_wanted]: total number of images to pull
[limit]: number of urls to write to each file

Image querying script written by Tamara Berg and extended heavily James Hays
http://graphics.cs.cmu.edu/projects/im2gps/flickr_code.html

Further modified by Daway Chou-Ren
"""


import sys, string, math, time, socket
from flickrapi import FlickrAPI
from datetime import datetime
from os.path import join
import os

import ipdb

socket.setdefaulttimeout(30)  #30 second time out on sockets before they throw
#an exception.  I've been having trouble with urllib.urlopen hanging in the
#flickr API.  This will show up as exceptions.IOError.

#the time out needs to be pretty long, it seems, because the flickr servers can be slow
#to respond to our big searches.

if len(sys.argv) != 6:
    print (__doc__)
    sys.exit(0)

print(sys.argv)
search_type = sys.argv[1]
query = sys.argv[2]
output_base_dir = sys.argv[3]
total_images_wanted = int(sys.argv[4])
limit = int(sys.argv[5])


''' Main search function. https://www.flickr.com/services/api/flickr.photos.search.html '''
def search_flickr(search_type, query, page_number, mintime, maxtime):
    if search_type == 'group':
        f = flickr.photos_search(media="photos", per_page="250", sort='relevance', page=str(page_number), group_id=query, min_upload_date=str(mintime), max_upload_date=str(maxtime))
    else:
        f = flickr.photos_search(media="photos", per_page="250", sort='relevance', page=str(page_number), text=query, min_upload_date=str(mintime), max_upload_date=str(maxtime))

    return f

def write_results(query, write_count, infos):
    query = query.replace(' ','_')
    output_dir = join(output_base_dir, query)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    outf_name = join(output_dir, str(write_count))
    with open(outf_name, 'w') as outf:
        for info in infos:
            outf.write(info + '\n')
        print('Wrote {} to {}'.format(len(infos), outf_name))
    wrote_query = True
    write_count += 1
    return wrote_query, write_count





'''
Does a binary search on search window size
Flickr API starts returning duplicates for API calls with 4000 results so we have to play with start/end dates to
ensure we do not fetch more than 4000 results per call.
'''
############### BEGIN ################

API_KEY = '639ee716be3441f34a88df4f92747883'
SECRET = 'b0cb0ba28e8d7a34'

flickr = FlickrAPI(API_KEY, SECRET)


# number of seconds to skip per query
#timeskip = 62899200 #two years
timeskip = 604800  #one week
# timeskip = 172800  #two days
#timeskip = 86400 #one day
#timeskip = 3600 #one hour
#timeskip = 2257 #for resuming previous query

mintime = 1325376000 # 1/1/2012
maxtime = mintime+timeskip
endtime =  1478131200  # 11/03/2016



infos = []
write_count = 0
query_count = 0
wrote_query = False

total_images_queried = 0


#this is the desired number of photos in each block
desired_photos = 250

print(datetime.fromtimestamp(mintime))
print(datetime.fromtimestamp(endtime))

while maxtime < endtime:
    #new approach - adjust maxtime until we get the desired number of images
    #within a block. We'll need to keep upper bounds and lower
    #lower bound is well defined (mintime), but upper bound is not. We can't
    #search all the way from endtime.

    lower_bound = mintime + 900 #lower bound OF the upper time limit. must be at least 15 minutes or zero results
    upper_bound = mintime + timeskip * 20 #upper bound of the upper time limit
    maxtime     = .95 * lower_bound + .05 * upper_bound

    print('\nBinary search on time range upper bound')
    print('Lower bound is ' + str(datetime.fromtimestamp(lower_bound)))
    print('Upper bound is ' + str(datetime.fromtimestamp(upper_bound)))

    keep_going = 6 #search stops after a fixed number of iterations
    while keep_going > 0 and maxtime < endtime:

        try:
            rsp = search_flickr(search_type, query, '1', mintime, maxtime)

            # ipdb.set_trace()
            time.sleep(1)
            total_images = rsp[0].get('total')
            null_test = int(total_images) #want to make sure this won't crash later on for some reason
            null_test = float(total_images)

            print('\nNum images: ' + total_images)
            print('Mintime: ' + str(mintime) + ' Maxtime: ' + str(maxtime) + ' Timeskip:  ' + str(maxtime - mintime))

            if int(total_images) > desired_photos:
                print('Too many photos in block, reducing maxtime')
                upper_bound = maxtime
                maxtime = (lower_bound + maxtime) / 2 #midpoint between current value and lower bound.

            if int(total_images) < desired_photos:
                print('Too few photos in block, increasing maxtime')
                lower_bound = maxtime
                maxtime = (upper_bound + maxtime) / 2

            print('Lower bound is ' + str(datetime.fromtimestamp(lower_bound)))
            print('Upper bound is ' + str(datetime.fromtimestamp(upper_bound)))

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
    print('Finished binary search')

    print('\nMintime: ' + str(mintime) + ' Maxtime: ' + str(maxtime))

    print('Num images: ' + total_images)

    current_image_num = 1

    num = int(rsp[0].get('pages'))
    print('Total pages: ' + str(num))

    #only visit 16 pages max, to try and avoid the dreaded duplicate bug
    #16 pages = 4000 images, should be duplicate safe.  Most interesting pictures will be taken.

    num_visit_pages = min(16,num)

    print('Visiting only ' + str(num_visit_pages) + ' pages (up to ' + str(num_visit_pages * 250) + ' images)')


    #print 'stopping before page ' + str(int(math.ceil(num/3) + 1)) + '\n'
    for page_number in range(0, num_visit_pages):
        f = search_flickr(search_type, query, page_number, mintime, maxtime)

        for i, photo in enumerate(f[0]):
            try:
                id = photo.get('id')
                # description = flickr.photos.getInfo(photo_id=id)[0][2].get('description')
                sizes = flickr.photos.getSizes(photo_id=id)[0]
                medium = sizes[5]  # can change this to get different sizes. https://www.flickr.com/services/api/flickr.photos.getSizes.html
                src_url = medium.get('source')
                infos.append(src_url)
                query_count += 1
            except:
                continue

            if query_count >= limit:
                write_results(query, write_count, infos)
                wrote_query = True
                write_count += 1
                total_images_queried += len(infos)
                if total_images_queried > total_images_wanted:
                    print ('Finished with {} images'.format(total_images_queried))
                    sys.exit(1)
                infos = []
                query_count = 0
                break

    timeskip = maxtime - mintime # used for initializing next binary search
    mintime  = maxtime


if not wrote_query:
    write_results(query, write_count, infos)
    total_images_queried += len(infos)
    wrote_query = True
    write_count += 1
    infos = []

print ('Finished with {} images'.format(total_images_queried))



