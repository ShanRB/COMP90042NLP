import json
import pprint


# read json
filename0 = "train.json"
file0 = open(filename0, 'r')
data = json.load(file0)
file0.close()

pp = pprint.PrettyPrinter(indent=4)
# preview data by line
toprint = [2]
for i in toprint:
    key = f'train-{i}'
    print(key)
    pp.pprint(data[key]['text'])

