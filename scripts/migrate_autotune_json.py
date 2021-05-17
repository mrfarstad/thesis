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
                        new_db[dimension][domain_dim][version][stencil_depth] = db[
                            dimension
                        ][domain_dim][version][stencil_depth]


# copy("results/results_autotune.json", "base")
# copy("results/results_autotune_0.json", "smem")
# copy("results/results_autotune_1.json", "smem_padded")
# copy("results/results_autotune.json", "smem_register")
copy("results/results_autotune.json")
copy("results/results_autotune_base.json", "base", "3")
copy("results/results_autotune_smem.json", "smem", "3")
copy("results/results_autotune_smem_padded.json", "smem_padded", "3")
copy("results/results_autotune_smem_register.json", "smem_register", "3")

# with open("results/results_autotune.json", "w") as fp:
#    json.dump(new_db, fp)

p.pprint(new_db)
