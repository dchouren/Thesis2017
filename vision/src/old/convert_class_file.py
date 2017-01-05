import sys
import json

class_file = sys.argv[1]
new_class_file = sys.argv[2]

new_class_mapping = {}
with open(class_file, 'r') as inf:
    for line in inf.readlines():
        line = line.strip()
        tokens = line.split(' ')
        if line[0] == '/':
            index = tokens[-1]
            hash = 'places_' + index
            label = tokens[0][3:]
        else:
            hash = tokens[0]
            index = tokens[-1]
            label = tokens[1].replace(',','')
        new_class_mapping[str(index)] = [str(hash), str(label)]

with open(new_class_file, 'w') as outf:
    json.dump(new_class_mapping, outf)
    
