import json
import math
import pprint as p
import subprocess

gpus = [2, 4]
depths = [1, 2, 4, 8, 16]

db = {}

for gpu in gpus:
    db[gpu] = {}

for gpu in gpus:
    for depth in depths:
        res = subprocess.run(
                ['./generate_halo_depth.sh',
                 'base',
                 str(gpu),
                 '32768',
                 #'256',
                 '32',
                 '32',
                 str(depth)],
                stdout=subprocess.PIPE).stdout.decode('utf-8')
        results = list(filter(None, res.split('\n')))
        db[gpu][depth] = [float(result) for result in results]

p.pprint(db)

with open('results_deep_halo.json', 'w') as fp:
    json.dump(db, fp)
