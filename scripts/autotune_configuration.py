import json
import math
import pprint as p
import subprocess
import sys
from functools import reduce


def deep_get(dictionary, keys, default=None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

dimensions = [2]
versions = ['base', 'smem', 'smem_prefetch']
unrolls = [1, 2, 4, 8, 16]
stencil_depths = [1, 2, 4, 8, 16]

try:
    with open('results/results_autotune.json', 'r') as jsonfile:
        db = json.load(jsonfile)
except FileNotFoundError:
    db = {}

for dimension in dimensions:
    if deep_get(db, str(dimension)) == None:
        db[str(dimension)] = {}
    dims = [8192]
    for dim in dims:
        if deep_get(db, ".".join([str(dimension), str(dim)])) == None:
            db[str(dimension)][str(dim)] = {}
        for version in versions:
            for unroll in unrolls:
                v = version
                if unroll > 1:
                    v += "_unroll_" + str(unroll)
                if deep_get(db, ".".join([str(dimension), str(dim), v])) == None:
                    db[str(dimension)][str(dim)][v] = {}
                for depth in stencil_depths:
                    if deep_get(db, ".".join([str(dimension), str(dim), v, str(depth)])) != None:
                        continue
                    db[str(dimension)][str(dim)][v][str(depth)] = {}
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
                    db[str(dimension)][str(dim)][v][str(depth)] = blockdims
                    with open('results.json', 'w') as fp:
                            json.dump(db, fp)

p.pprint(db)
