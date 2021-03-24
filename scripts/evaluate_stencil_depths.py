import json
import math
import pprint as p
import subprocess

#dimensions = [2, 3]
dimensions = [2]
versions = ['base', 'base_unroll_2', 'smem', 'smem_unroll_4', 'smem_prefetch_unroll_4']
stencil_depths = [1, 2, 4, 8, 16, 32]
#versions = ['smem_prefetch_unroll_4']
#stencil_depths = [16]

db = {}
for dimension in dimensions:
    db[dimension] = {}
    if dimension == 3:
        dims = [256, 512, 1024]
    else:
        dims = [8192, 16384, 32768]
    for dim in dims:
        db[dimension][dim] = {}
        for version in versions:
            db[dimension][dim][version] = {}
            for depth in stencil_depths:
                db[dimension][dim][version][depth] = {}
            for depth in stencil_depths:
                res = subprocess.run(
                        ['./scripts/evaluate_configuration.sh',
                         version,
                         '1',
                         str(dim),
                         str(dimension),
                         '32',
                         '32',
                         '1',
                         str(depth),
                         '5'],
                        stdout=subprocess.PIPE).stdout.decode('utf-8')
                results = list(filter(None, res.split('\n')))
                db[dimension][dim][version][depth] = [float(result) for result in results]
p.pprint(db)
with open('results.json', 'w') as fp:
    json.dump(db, fp)
