import json
import pprint as p
import sys
from functools import reduce

new_db = {}

def deep_get(dictionary, keys, default=None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

def entry_not_exists(db, nested_list):
    return deep_get(db, ".".join(nested_list)) == None

def copy(config, results_json, host):
    with open(results_json) as file: 
        db = json.loads(file.read())
        for dimension, dimension_db in db.items():
            if entry_not_exists(new_db, [dimension]):
                new_db[dimension] = {}
            for domain_dim, domain_dim_db in dimension_db.items():
                if entry_not_exists(new_db, [dimension, domain_dim]):
                    new_db[dimension][domain_dim] = {}
                for version, version_db in domain_dim_db.items():
                    version = version.replace("smem_prefetch", "smem_padded")
                    if entry_not_exists(new_db, [dimension, domain_dim, version]):
                        new_db[dimension][domain_dim][version] = {}
                    for stencil_depth, stencil_depth_db in version_db.items():
                        if entry_not_exists(new_db, [dimension, domain_dim, version, stencil_depth]):
                            new_db[dimension][domain_dim][version][stencil_depth] = {}
                        for iteration, iteration_db in stencil_depth_db.items():
                            if entry_not_exists(new_db, [dimension, domain_dim, version, stencil_depth, iteration]):
                                new_db[dimension][domain_dim][version][stencil_depth][iteration] = {}
                            if entry_not_exists(new_db, [dimension, domain_dim, version, stencil_depth, iteration, host]):
                                new_db[dimension][domain_dim][version][stencil_depth][iteration][host] = {}
                            if entry_not_exists(db, [dimension, domain_dim, version, stencil_depth, iteration, host, config]):
                                continue
                            new_db[dimension][domain_dim][version][stencil_depth][iteration][host][config] = iteration_db[host][config]


copy("heuristic", "results/results_stencil_depths_heuristic.json", "heid")
copy("autotune", "results/results_stencil_depths_autotuned.json", "heid")
copy("heuristic", "results/results_stencil_depths_idun.json", "idun")

#with open("results/results_stencil_depths_autotuned.json", 'w') as fp:
#with open("results/results_stencil_depths_idun.json", 'w') as fp:
with open("results/results_stencil_depths.json", 'w') as fp:
    json.dump(new_db, fp)

p.pprint(new_db)
