import json
import math
import pprint as p
import subprocess

#dimensions = [2, 3]
dimensions = [2]
unroll_x = [1, 2, 4, 8]
stencil_depths = [1, 2, 4, 8, 16, 32, 64]

db = {}
for dimension in dimensions:
    db[dimension] = {}
    if dimension == 3:
        dims = [256, 512, 1024]
    else:
        #dims = [16384, 32768, 65536]
        dims = [8192, 16384, 32768]
    for dim in dims:
        db[dimension][dim] = {}
        for depth in stencil_depths:
            db[dimension][dim][depth] = {}
        for depth in stencil_depths:
            for ux in unroll_x:
                db[dimension][dim][depth][ux] = {}
                res = subprocess.run(
                        ['./scripts/evaluate_configuration.sh',
                         'base',
                         '1',
                         str(dim),
                         str(dimension),
                         '16',
                         '8',
                         '8',
                         str(depth),
                         '5',
                         str(ux)],
                        stdout=subprocess.PIPE).stdout.decode('utf-8')
                results = list(filter(None, res.split('\n')))
                db[dimension][dim][depth][ux] = [float(result) for result in results]
p.pprint(db)
with open('results.json', 'w') as fp:
    json.dump(db, fp)
