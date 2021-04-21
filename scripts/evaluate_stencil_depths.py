import json
import math
import pprint as p
import subprocess
import sys
from functools import reduce

dimensions = ['2']
iterations = ['8', '1024']
versions = ['base', 'smem', 'smem_prefetch']
stencil_depths = ['1', '2', '4', '8', '16']
unrolls = ['1', '2', '4', '8']
host = "heid"
gpus = ['1', '2', '4', '8', '16']
if len(sys.argv) > 1 and sys.argv[1] == "True": 
    autotune = True
else:
    autotune = False
config = 'autotune' if autotune else 'heuristic'

# TODO: Run autotuned executions for unroll factors 1-8 for base, smem, smem_prefetch
if autotune:
    try:
        with open('results/results_autotune.json', 'r') as jsonfile:
            tune_db = json.load(jsonfile)
    except FileNotFoundError:
        print("Autotune file not found!")
        autotune = False
        tune_db = {}
else:
    tune_db = {}

def deep_get(dictionary, keys, default=None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

def entry_exists(nested_list):
    return deep_get(db, ".".join(list(map(str,nested_list)))) != None

def autotune_entry_exists(nested_list):
    return deep_get(tune_db, ".".join(list(map(str,nested_list)))) != None

try:
    with open('results/results_stencil_depths.json', 'r') as jsonfile:
        db = json.load(jsonfile)
except FileNotFoundError:
    print("Stencil depth file not found!")
    sys.exit()
    
for dimension in dimensions:
    if not entry_exists([dimension]):
        db[dimension] = {}
    if dimension == 3:
        dims = ['256', '512', '1024']
    else:
        dims = ['8192', '32768']
    for dim in dims:
        if not entry_exists([dimension, dim]):
            db[dimension][dim] = {}
        for gpu in gpus:
            if gpu > "1":
                versions = ["base"]
                unrolls = ["1"]
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
                        for iteration in iterations:
                            if not entry_exists([dimension, dim, v, depth, iteration]):
                                db[dimension][dim][v][depth][iteration] = {}
                            if not entry_exists([dimension, dim, v, depth, iteration, host]):
                                db[dimension][dim][v][depth][iteration][host] = {}
                            if entry_exists([dimension, dim, v, depth, iteration, host, config]):
                                continue
                            if autotune_entry_exists([dimension, dim, v_tune, depth]):
                                blockdims = tune_db[dimension][dim][v_tune][depth]
                            res = subprocess.run(
                                    ['./scripts/evaluate_configuration.sh',
                                     version,
                                     gpu,
                                     dim,
                                     dimension,
                                     '32' if not autotune else str(blockdims['BLOCK_X']),
                                     '32' if not autotune else str(blockdims['BLOCK_Y']),
                                     '1',
                                     depth,
                                     '30',
                                     '0',
                                     unroll,
                                     iteration],
                                    stdout=subprocess.PIPE).stdout.decode('utf-8')
                            results = list(map(float,filter(lambda s: not "declare" in s, filter(None, res.split('\n')))))
                            db[dimension][dim][v][depth][iteration][host][config] = results
                            with open("results.json", 'w') as fp:
                                json.dump(db, fp)

with open("results.json", 'w') as fp:
    json.dump(db, fp)
p.pprint(db)
