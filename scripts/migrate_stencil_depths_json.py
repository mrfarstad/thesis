import json
import pprint as p
from functools import reduce

new_db = {}

def deep_get(dictionary, keys, default=None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

def entry_not_exists(nested_list):
    return deep_get(new_db, ".".join(nested_list)) == None

def copy(config, results_json):
    with open(results_json) as file: 
        db = json.loads(file.read())
        for dimension, dimension_db in db.items():
            if entry_not_exists([dimension]):
                new_db[dimension] = {}
            for domain_dim, domain_dim_db in dimension_db.items():
                if entry_not_exists([dimension, domain_dim]):
                    new_db[dimension][domain_dim] = {}
                for version, version_db in domain_dim_db.items():
                    if entry_not_exists([dimension, domain_dim, version]):
                        new_db[dimension][domain_dim][version] = {}
                    for stencil_depth, times_db in version_db.items():
                        if entry_not_exists([dimension, domain_dim, version, stencil_depth]):
                            new_db[dimension][domain_dim][version][stencil_depth] = {}
                        if entry_not_exists([dimension, domain_dim, version, stencil_depth, "8"]):
                            new_db[dimension][domain_dim][version][stencil_depth]["8"] = {}
                        for execution_config, times in times_db.items():
                            if execution_config == config:
                                new_db[dimension][domain_dim][version][stencil_depth]["8"][config] = times

copy("heuristic", "results/results_stencil_depths_heuristic.json")
copy("autotune", "results/results_stencil_depths_autotuned.json")

with open('results/results_stencil_depths.json', 'w') as fp:
    json.dump(new_db, fp)

p.pprint(new_db)
