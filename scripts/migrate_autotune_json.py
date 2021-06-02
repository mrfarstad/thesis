import json
import pprint as p
import sys
from functools import reduce

new_db = {}


def deep_get(dictionary, keys, default=None):
    return reduce(
        lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
        keys.split("."),
        dictionary,
    )


def entry_not_exists(db, nested_list):
    return deep_get(db, ".".join(nested_list)) == None


def copy(results_json, v="all", dim="2"):
    with open(results_json) as file:
        db = json.loads(file.read())
        for dimension, dimension_db in db.items():
            if dimension != dim:
                continue
            if entry_not_exists(new_db, [dimension]):
                new_db[dimension] = {}
            for domain_dim, domain_dim_db in dimension_db.items():
                if entry_not_exists(new_db, [dimension, domain_dim]):
                    new_db[dimension][domain_dim] = {}
                for version, version_db in domain_dim_db.items():
                    if v != "all" and (not (version == v or v + "_unroll" in version)):
                        continue
                    if entry_not_exists(new_db, [dimension, domain_dim, version]):
                        new_db[dimension][domain_dim][version] = {}
                    for stencil_depth, stencil_depth_db in version_db.items():
                        if dim == "3" and int(stencil_depth) > 4:
                            continue
                        new_db[dimension][domain_dim][version][stencil_depth] = {
                            k: v
                            for k, v in db[dimension][domain_dim][version][
                                stencil_depth
                            ].items()
                            # Handle case where HEURISTIC gets stored in json file
                            if not "HEURISTIC" in k
                        }


copy("results/results_autotune_base_3d.json", "base", "3")
copy("results/results_autotune_smem_3d.json", "smem", "3")
copy("results/results_autotune_smem_padded_3d.json", "smem_padded", "3")

with open("results/results_autotune.json", "w") as fp:
    json.dump(new_db, fp)

p.pprint(new_db)
