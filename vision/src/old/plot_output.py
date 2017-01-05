import sys
import glob
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

import ipdb

pred_dir = join('output',sys.argv[1])
pred_files = glob.glob(join(pred_dir, '*'))

max_probs = []
pred_counts = {'food': 0, 'drink': 0, 'inside': 0, 'outside': 0, 'menu': 0}
pred_sum_probs = {'food': 0.0, 'drink': 0.0, 'inside': 0.0, 'outside': 0.0, 'menu': 0.0}

menu_output = ""
location_menu_counts = {}
locations = []
location_count = 0

for pred_file in pred_files:
    with open(pred_file, 'r') as inf:
        next(inf)
        for pred in inf:
            mediaid, locationid, url, prediction, p1, p2, p3, p4, p5 = pred.split(',')
            max_class_prob = max([float(p1), float(p2), float(p3), float(p4), float(p5)])
            max_probs.append(max_class_prob)
            pred_counts[prediction] += 1
            pred_sum_probs[prediction] += max_class_prob

            if prediction == 'menu':
                menu_output += pred

                try:
                    location_menu_counts[locationid] += 1
                except KeyError:
                    location_menu_counts[locationid] = 1

            if not locationid in locations:
                locations.append(locationid)
                location_count += 1

max_probs = np.asarray(max_probs)
total = len(max_probs)

bins = np.arange(0.0,1.1,0.1)
hist = np.histogram(max_probs, bins)

# ipdb.set_trace()
plt.title("Best Class Probability Hist  N="+str(total))
plt.ylabel("Count")
plt.xlabel("Predicted Probability of Best Class")
plt.grid()
x = plt.hist(max_probs, histtype='step', bins=bins)
plt.savefig("plots/best_class_probability_hist")

plt.clf()

cum_probs = np.cumsum(hist[0])
cum_probs_p = cum_probs / sum(hist[0])
# ipdb.set_trace()
y = plt.plot(bins[1:], 1-cum_probs_p)
plt.title("Best Class Probability vs Coverage  N="+str(total))
plt.xlabel("Coverage %")
plt.xticks(np.arange(0.1,1.1,0.1))
plt.ylabel("Predicted Probability of Best Class")
plt.yticks(np.arange(0.0,1.2,0.1))
plt.grid()
plt.savefig("plots/predicted_probability_v_coverage")


# for key, value in pred_counts.items():
#     pred_counts[key] = value / total
print(pred_counts)
print(pred_sum_probs)

avg_pred_probs = {}

for key in pred_sum_probs.keys():
    avg_pred_probs[key] = pred_sum_probs[key] / pred_counts[key]

print(avg_pred_probs)
print("Num photos: " + str(total))

print("Num locations: " + str(location_count))
num_with_menu = len(location_menu_counts.keys())
total_menu_photos = 0
for location, count in location_menu_counts.items():
    total_menu_photos += count
print("Num menu photos: " + str(total_menu_photos))
print("Num locations with menu photos: " + str(num_with_menu))
# print(location_menu_counts)

ipdb.set_trace()