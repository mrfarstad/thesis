import json
import math
import pprint as p
import subprocess
import sys

dimensions = [2]
#versions = ['base', 'smem', 'smem_prefetch']
versions = ['base']
#stencil_depths = [1, 2, 4, 8, 16]
stencil_depths = [1, 16]
unrolls = [1, 2, 4]

db = {}
for dimension in dimensions:
    db[dimension] = {}
    dims = [8192]
    for dim in dims:
        db[dimension][dim] = {}
        for version in versions:
            for unroll in unrolls:
                v = version
                if unroll > 1:
                    v += "_unroll_" + str(unroll)
                db[dimension][dim][v] = {}
                for depth in stencil_depths:
                    db[dimension][dim][v][depth] = {}
                for depth in stencil_depths:
                    res = subprocess.run(
                            ['./scripts/autotune_configuration.sh',
                             version,
                             '1',
                             str(dim),
                             str(dimension),
                             str(depth),
                             '5',
                             '0',
                             str(unroll)],
                            stdout=subprocess.PIPE).stdout.decode('utf-8')
                    results = list(filter(None, res.split('\n')))
                    blockdims = results[1].split(',')
                    blockdims.pop()
                    blockdims = [b.strip() for b in blockdims]
                    blockdims = [b.split(' = ') for b in blockdims]
                    blockdims = {b[0]: int(b[1]) for b in blockdims}
                    db[dimension][dim][v][depth] = blockdims
                    with open('autotune.json', 'w') as fp:
                        json.dump(db, fp)

p.pprint(db)
