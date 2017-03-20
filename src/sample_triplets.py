import sys
from geopy.distance import vincenty



h5_dir = sys.argv[1]


def meter_distance(geo_loc1, geo_loc2):
    return vincenty(geo_loc1, geo_loc2).meters
