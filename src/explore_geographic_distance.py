import numpy as np
from geopy.distance import vincenty
import itertools
import time
import hickle
import pickle
import matplotlib.pyplot as plt

import ipdb

csv_path = './resources/2014_geo'


with open(csv_path, 'r') as inf:
    geo_locs = np.loadtxt(inf, delimiter=',')


def meter_distance(geo_loc1, geo_loc2):
    return vincenty(geo_loc1, geo_loc2).meters

random_geo_locs = geo_locs[np.random.choice(geo_locs.shape[0], 10000)]

# ipdb.set_trace()

distances = []
for i, geo_loc_pair in enumerate(itertools.combinations(random_geo_locs, 2)):
    if i % 1000 == 0:
        # print(i, time.time())
        distance = meter_distance(*geo_loc_pair)
        distances.append(distance)

# distances = [meter_distance(*geo_loc_pair) for geo_loc_pair in itertools.combinations(random_geo_locs, 2)]

with open('2014_random_distances.hkl', 'wb') as outf:
    pickle.dump(distances, outf)

plt.title('Pairwise Distances')
plt.xlabel('Distance (m)')
plt.ylabel('Frequency')
plt.hist(distances)
plt.savefig('plots/2014_distances.png')
plt.show()

plt.title('Pairwise Distances < 5000m')
plt.xlabel('Distance (m)')
plt.ylabel('Frequency')
plt.hist(distances, bins=[0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
plt.savefig('plots/2014_distances_5000.png')
plt.show()

plt.title('Pairwise Distances < 2000m')
plt.xlabel('Distance (m)')
plt.ylabel('Frequency')
plt.hist(distances, bins=[0,200,400,600,800,1000,1200,1400,1600,1800,2000])
plt.savefig('plots/2014_distances_2000.png')
plt.show()

plt.title('Pairwise Distances < 100m')
plt.xlabel('Distance (m)')
plt.ylabel('Frequency')
plt.hist(distances, bins=[0,10,20,30,40,50,60,70,80,90,100])
plt.savefig('plots/2014_distances_100.png')
plt.show()

plt.title('Pairwise Distances < 10m')
plt.xlabel('Distance (m)')
plt.ylabel('Frequency')
plt.hist(distances, bins=[0,1,2,3,4,5,6,7,8,9,10])
plt.savefig('plots/2014_distances_10.png')
plt.show()





ipdb.set_trace()
























