import json
import math
import pprint as p
import subprocess

dimensions = [2, 3]
smem_pad = [0, 1]
stencil_depths = [1, 2, 4, 8, 16, 32, 64]

db = {}
for dimension in dimensions:
    db[dimension] = {}
    if dimension == 3:
        dims = [256, 512, 1024]
    else:
        dims = [8192, 16384, 32768]
    for dim in dims:
        db[dimension][dim] = {}
        for depth in stencil_depths:
            db[dimension][dim][depth] = {}
        for depth in stencil_depths:
            for pad in smem_pad:
                db[dimension][dim][depth][pad] = {}
                res = subprocess.run(
                        ['./scripts/evaluate_configuration.sh',
                         'base',
                         '1',
                         str(dim),
                         str(dimension),
                         '32',
                         '16',
                         '2',
                         str(depth),
                         '5',
                         str(pad),
                         '1'],
                        stdout=subprocess.PIPE).stdout.decode('utf-8')
                results = list(filter(None, res.split('\n')))
                db[dimension][dim][depth][pad] = [float(result) for result in results]
p.pprint(db)
with open('results.json', 'w') as fp:
    json.dump(db, fp)
