import json
import math
import pprint as p
import subprocess

dimensions = [2]
versions = ['base', 'smem', 'smem_prefetch']
stencil_depths = [1, 2, 4, 8, 16]

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
            if version == 'base':
                unrolls = [1, 2]
            elif version == 'smem':
                unrolls = [1, 4]
            for unroll in unrolls:
                v = version
                if unroll > 1:
                    v += + "_unroll_" + str(unroll)
                db[dimension][dim][v] = {}
                for depth in stencil_depths:
                    db[dimension][dim][v][depth] = {}
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
                             '20',
                             '0',
                             str(unroll)],
                            stdout=subprocess.PIPE).stdout.decode('utf-8')
                    results = list(filter(None, res.split('\n')))
                    db[dimension][dim][v][depth] = [float(result) for result in results]
p.pprint(db)
with open('results.json', 'w') as fp:
    json.dump(db, fp)
