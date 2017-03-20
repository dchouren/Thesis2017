from geopy.distance import vincenty

def meter_distance(geo_loc1, geo_loc2):
    return vincenty(geo_loc1, geo_loc2).meters
