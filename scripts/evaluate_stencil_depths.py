import json
import math
import pprint as p
import subprocess

gpus = [1, 2, 4]
stencil_depths = [1, 2, 4, 8, 16, 32]

db = {}

for gpu in gpus:
    db[gpu] = {}

for gpu in gpus:
    for depth in stencil_depths:
        res = subprocess.run(
                ['./scripts/evaluate_stencil_depths.sh',
                 'base',
                 str(gpu),
                 '256',
                 '32',
                 '8',
                 '4',
                 str(depth)],
                stdout=subprocess.PIPE).stdout.decode('utf-8')
        results = list(filter(None, res.split('\n')))
        db[gpu][depth] = [float(result) for result in results]

p.pprint(db)

with open('results_stencil_depths.json', 'w') as fp:
    json.dump(db, fp)
