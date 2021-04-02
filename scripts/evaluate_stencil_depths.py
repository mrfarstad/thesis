import json
import math
import pprint as p
import subprocess
from functools import reduce

dimensions = [2]
versions = ['base', 'smem', 'smem_prefetch']
stencil_depths = [1, 2, 4, 8, 16]
gpus = [1, 2, 4, 8, 16]
autotune = False

def deep_get(dictionary, keys, default=None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

#if autotune:
#    try:
#        with open('results/results_autotune.json', 'r') as jsonfile:
#            tune_db = json.load(jsonfile)
#    except FileNotFoundError:
#        print("Autotune file not found!")
#        autotune = False
#        tune_db = {}
#else:
#    tune_db = {}

try:
    with open('results/results_stencil_depths.json', 'r') as jsonfile:
        db = json.load(jsonfile)
except FileNotFoundError:
    print("Stencil depth file not found!")
    db = {}
    
for dimension in dimensions:
    if deep_get(db, str(dimension)) == None:
        db[str(dimension)] = {}
    if dimension == 3:
        dims = [256, 512, 1024]
    else:
        dims = [8192, 32768]
        #dims = [8192, 16384, 32768]
    for dim in dims:
        if deep_get(db, ".".join([str(dimension), str(dim)])) == None:
            db[str(dimension)][str(dim)] = {}
        for gpu in gpus:
            for version in versions:
                if version == 'base':
                    #unrolls = [1, 2]
                    unrolls = [1]
                elif 'smem' in version:
                    #unrolls = [1, 4]
                    unrolls = [1]
                for unroll in unrolls:
                    v0 = str(gpu) + "_gpus_" if gpu > 0 else "_gpu_"
                    v = v0 + version
                    if unroll > 1:
                        v += "_unroll_" + str(unroll)
                    v_tune = v[len(v0):]
                    if deep_get(db, ".".join([str(dimension), str(dim), v])) == None:
                        db[str(dimension)][str(dim)][v] = {}
                    for depth in stencil_depths:
                        if deep_get(db, ".".join([str(dimension), str(dim), v, str(depth)])) != None:
                            continue
                        #if deep_get(tune_db, ".".join([str(dimension), str(8192), v_tune, str(depth)])) != None:
                        #    blockdims = tune_db[str(dimension)][str(8192)][v_tune][str(depth)]
                        res = subprocess.run(
                                ['./scripts/evaluate_configuration.sh',
                                 version,
                                 str(gpu),
                                 str(dim),
                                 str(dimension),
                                 #'32' if not autotune else str(blockdims['BLOCK_X']),
                                 #'32' if not autotune else str(blockdims['BLOCK_Y']),
                                 '32',
                                 '32',
                                 '1',
                                 str(depth),
                                 '5',
                                 '0',
                                 str(unroll)],
                                stdout=subprocess.PIPE).stdout.decode('utf-8')
                        results = list(filter(None, res.split('\n')))
                        db[str(dimension)][str(dim)][v][str(depth)] = [float(result) for result in results]
                        with open('results.json', 'w') as fp:
                            json.dump(db, fp)

with open('results.json', 'w') as fp:
    json.dump(db, fp)
p.pprint(db)
