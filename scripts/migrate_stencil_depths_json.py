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


def copy(results_json, host, config):
    with open(results_json) as file:
        db = json.loads(file.read())
        for dimension, dimension_db in db.items():
            if "3d" in results_json and not dimension == "3":
                continue
            if entry_not_exists(new_db, [dimension]):
                new_db[dimension] = {}
            for domain_dim, domain_dim_db in dimension_db.items():
                if entry_not_exists(new_db, [dimension, domain_dim]):
                    new_db[dimension][domain_dim] = {}

                v_found = None
                for v in domain_dim_db.keys():
                    if v.replace("1_gpus_", "") in results_json:
                        v_found = v
                for version, version_db in domain_dim_db.items():
                    if v_found is not None and not v_found in version:
                        continue
                    if entry_not_exists(new_db, [dimension, domain_dim, version]):
                        new_db[dimension][domain_dim][version] = {}
                    for stencil_depth, stencil_depth_db in version_db.items():
                        if entry_not_exists(
                            new_db, [dimension, domain_dim, version, stencil_depth]
                        ):
                            new_db[dimension][domain_dim][version][stencil_depth] = {}
                        for iteration, iteration_db in stencil_depth_db.items():
                            if config == "autotune" or host == "idun":
                                if not (iteration == "8" and "1_gpus" in version):
                                    continue
                                if (
                                    config == "autotune"
                                    # and "register" in version
                                    and not any(
                                        domain_dim == d for d in ["1024", "32768"]
                                    )
                                ):
                                    continue
                            if "profile" in results_json:
                                if not (
                                    iteration == "8"
                                    and "1_gpus" in version
                                    and any(domain_dim == d for d in ["1024", "32768"])
                                ):
                                    continue
                            if entry_not_exists(
                                new_db,
                                [
                                    dimension,
                                    domain_dim,
                                    version,
                                    stencil_depth,
                                    iteration,
                                ],
                            ):
                                new_db[dimension][domain_dim][version][stencil_depth][
                                    iteration
                                ] = {}
                            if entry_not_exists(
                                new_db,
                                [
                                    dimension,
                                    domain_dim,
                                    version,
                                    stencil_depth,
                                    iteration,
                                    host,
                                ],
                            ):
                                new_db[dimension][domain_dim][version][stencil_depth][
                                    iteration
                                ][host] = {}
                            if entry_not_exists(
                                new_db,
                                [
                                    dimension,
                                    domain_dim,
                                    version,
                                    stencil_depth,
                                    iteration,
                                    host,
                                    config,
                                ],
                            ):
                                new_db[dimension][domain_dim][version][stencil_depth][
                                    iteration
                                ][host][config] = {}
                            for metric, times in iteration_db[host][config].items():
                                if not "profile" in results_json and metric != "time":
                                    continue
                                if "profile" in results_json and metric == "time":
                                    continue
                                if entry_not_exists(
                                    new_db,
                                    [
                                        dimension,
                                        domain_dim,
                                        version,
                                        stencil_depth,
                                        iteration,
                                        host,
                                        config,
                                        metric,
                                    ],
                                ):
                                    new_db[dimension][domain_dim][version][
                                        stencil_depth
                                    ][iteration][host][config][metric] = {}
                                new_db[dimension][domain_dim][version][stencil_depth][
                                    iteration
                                ][host][config][metric] = iteration_db[host][config][
                                    metric
                                ]


# copy("results/results_batch_profile_autotune.json", "heid", "autotune")
# copy("results/results_stencil_depths_heuristic.json", "heid", "heuristic")
# TODO: Find out why smem_padded fails so hard for autotuning in 3D
# copy("results/results_stencil_depths_autotuned.json", "heid", "autotune") # 3D autotune is not present in this file

# copy("results/results_batch_profile.json", "heid", "heuristic")
copy("results/results_batch_profile_base_3d.json", "heid", "heuristic")
copy("results/results_batch_profile_smem_3d.json", "heid", "heuristic")
copy("results/results_batch_profile_smem_padded_3d.json", "heid", "heuristic")

copy("results/results_stencil_depths_heuristic_base_3d.json", "heid", "heuristic")
copy("results/results_stencil_depths_heuristic_smem_3d.json", "heid", "heuristic")
copy(
    "results/results_stencil_depths_heuristic_smem_padded_3d.json", "heid", "heuristic"
)
# copy("results/results_stencil_depths_idun.json", "idun", "heuristic")

# copy("results/results_stencil_depths_autotuned_base_3d.json", "heid", "autotune")
# copy("results/results_stencil_depths_autotuned_smem_3d.json", "heid", "autotune")
# copy("results/results_stencil_depths_autotuned_smem_3d.json", "heid", "autotune")

# with open("results/results_stencil_depths_heuristic.json", 'w') as fp:
# with open("results/results_stencil_depths_autotuned.json", 'w') as fp:
# with open("results/results_stencil_depths_idun.json", 'w') as fp:
with open("results/results_stencil_depths.json", "w") as fp:
    json.dump(new_db, fp)

p.pprint(new_db)
