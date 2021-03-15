import json
import math
import pprint as p
import subprocess

gpus = [1, 2, 4, 8]
stencil_depths = [1, 2, 4, 8, 16, 32, 64, 128]

db = {}
for gpu in gpus:
    db[gpu] = {}
for gpu in gpus:
    if gpu == 1:
        versions = ['base', 'smem', 'coop']#, 'coop_smem']
    else:
        versions = ['base', 'smem']
    for version in versions:
        db[gpu][version] = {}
        for depth in stencil_depths:
            res = subprocess.run(
                    ['./scripts/evaluate_stencil_depths.sh',
                     version,
                     str(gpu),
                     '1024',
                     '16',
                     '8',
                     '8',
                     str(depth),
                     '5'],
                    stdout=subprocess.PIPE).stdout.decode('utf-8')
            results = list(filter(None, res.split('\n')))
            db[gpu][version][depth] = [float(result) for result in results]
p.pprint(db)
with open('results_stencil_depths.json', 'w') as fp:
    json.dump(db, fp)
