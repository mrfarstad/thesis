import json
import math
import pprint as p
import subprocess
import sys
from functools import reduce


def deep_get(dictionary, keys, default=None):
    return reduce(
        lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
        keys.split("."),
        dictionary,
    )


def entry_exists(nested_list):
    return deep_get(db, ".".join(list(map(str, nested_list)))) != None


dimensions = ["2"]
versions = ["base", "smem", "smem_padded", "smem_register"]
unrolls = ["1", "2", "4", "8"]
stencil_depths = ["1", "2", "4", "8", "16"]

try:
    with open("results/results_autotune.json", "r") as jsonfile:
        db = json.load(jsonfile)
except FileNotFoundError:
    db = {}

for dimension in dimensions:
    if not entry_exists([dimension]):
        db[dimension] = {}
    # dims = ["8192", "32768"]
    dims = ["32768"]
    for dim in dims:
        if not entry_exists([dimension, dim]):
            db[dimension][dim] = {}
        for version in versions:
            for unroll in unrolls:
                v = version
                if int(unroll) > 1:
                    v += "_unroll_" + unroll
                if not entry_exists([dimension, dim, v]):
                    db[dimension][dim][v] = {}
                for depth in stencil_depths:
                    if entry_exists([dimension, dim, v, depth]):
                        continue
                    db[dimension][dim][v][depth] = {}
                    res = subprocess.run(
                        [
                            "./scripts/autotune_configuration.sh",
                            version,
                            "1",
                            dim,
                            dimension,
                            depth,
                            "30",
                            "0",
                            unroll,
                            "8",
                        ],
                        stdout=subprocess.PIPE,
                    ).stdout.decode("utf-8")
                    results = list(filter(None, res.split("\n")))
                    blockdims = results[1].split(",")
                    blockdims.pop()
                    blockdims = [b.strip() for b in blockdims]
                    blockdims = [b.split(" = ") for b in blockdims]
                    blockdims = {b[0]: int(b[1]) for b in blockdims}
                    db[dimension][dim][v][depth] = blockdims
                    with open("results.json", "w") as fp:
                        json.dump(db, fp)

with open("results.json", "w") as fp:
    json.dump(db, fp)
p.pprint(db)
