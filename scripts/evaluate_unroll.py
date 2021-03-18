import json
import math
import pprint as p
import subprocess

dims = [256, 512, 1024]
unroll_x = [1, 2, 4, 8]
stencil_depths = [1, 2, 4, 8, 16, 32, 64]

db = {}
for dim in dims:
    db[dim] = {}
    for depth in stencil_depths:
        db[dim][depth] = {}
    for depth in stencil_depths:
        for ux in unroll_x:
            db[dim][depth][ux] = {}
            res = subprocess.run(
                    ['./scripts/evaluate_configuration.sh',
                     'base',
                     '1',
                     str(dim),
                     '16',
                     '8',
                     '8',
                     str(depth),
                     '5',
                     str(ux)],
                    stdout=subprocess.PIPE).stdout.decode('utf-8')
            results = list(filter(None, res.split('\n')))
            db[dim][depth][ux] = [float(result) for result in results]
p.pprint(db)
with open('results.json', 'w') as fp:
    json.dump(db, fp)
