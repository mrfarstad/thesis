import json
import math
import pprint as p
import subprocess
from functools import reduce

dimensions = ['2']
versions = ['base', 'smem', 'smem_prefetch']
stencil_depths = ['1', '2', '4', '8', '16']
#gpus = ['1', '2', '4', '8', '16']
gpus = ['1']
autotune = False
unrolls = ['1', '2', '4', '8']

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

def deep_get(dictionary, keys, default=None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

def entry_exists(nested_list):
    return deep_get(db, ".".join(list(map(str,nested_list)))) != None

try:
    with open('results/results_stencil_depths_migrated.json', 'r') as jsonfile:
        db = json.load(jsonfile)
except FileNotFoundError:
    print("Stencil depth file not found!")
    db = {}
    
for dimension in dimensions:
    if not entry_exists([dimension]):
        db[dimension] = {}
    if dimension == 3:
        dims = [256, 512, 1024]
    else:
        dims = [8192, 32768]
    for dim in dims:
        if not entry_exists([dimension, dim]):
            db[dimension][dim] = {}
        for gpu in gpus:
            for version in versions:
                for unroll in unrolls:
                    v0 = gpu + "_gpus_" if int(gpu) > 0 else "_gpu_"
                    v = v0 + version
                    if int(unroll) > 1:
                        v += "_unroll_" + unroll
                    v_tune = v[len(v0):]
                    if not entry_exists([dimension, dim, v]):
                        db[dimension][dim][v] = {}
                    for depth in stencil_depths:
                        if not entry_exists([dimension, dim, v, depth]):
                            db[dimension][dim][v][depth] = {}
                        if entry_exists([dimension, dim, v, depth, 'heuristic']):
                            continue
                        #if deep_get(tune_db, ".".join([str(dimension), str(8192), v_tune, str(depth)])) != None:
                        #    blockdims = tune_db[str(dimension)][str(8192)][v_tune][str(depth)]
                        res = subprocess.run(
                                ['./scripts/evaluate_configuration.sh',
                                 version,
                                 gpu,
                                 dim,
                                 dimension,
                                 #'32' if not autotune else str(blockdims['BLOCK_X']),
                                 #'32' if not autotune else str(blockdims['BLOCK_Y']),
                                 '32',
                                 '32',
                                 '1',
                                 depth,
                                 '5',
                                 '0',
                                 unroll],
                                stdout=subprocess.PIPE).stdout.decode('utf-8')

                        results = list(filter(None, res.split('\n')))
                        db[dimension][dim][v][depth]['heuristic'] = [result for result in results]
                        with open('results.json', 'w') as fp:
                            json.dump(db, fp)

with open('results.json', 'w') as fp:
    json.dump(db, fp)
p.pprint(db)
