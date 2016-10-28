import flickrapi
import ipdb

flickrAPIKey = "2b601611576cb460560f1c11cf6cbb45"  # API key
flickrSecret = "654a32a8f9ce451d"                  # shared "secret"

flickr = flickrapi.FlickrAPI(flickrAPIKey, flickrSecret)


query_string = 'food'

timeskip = 172800  #two days
mintime = 1171416400 #resume crash WashingtonDC
maxtime = mintime+timeskip
endtime =  1192165200 

photos = flickr.photos.search(api_key=flickrAPIKey, ispublic="1", media="photos", per_page="250", page="1", has_geo = "1", #bbox="-180, -90, 180, 90",text=query_string,accuracy="6", #6 is region level.  most things seem 10 or better.min_upload_date=str(mintime),max_upload_date=str(maxtime))

ipdb.set_trace()
# for photo in flickr.walk(tag_mode='all',
#         tags='sybren,365,threesixtyfive',
#         min_taken_date='2008-08-20',
#         max_taken_date='2008-08-30'):
#     print (photo.get('title'))