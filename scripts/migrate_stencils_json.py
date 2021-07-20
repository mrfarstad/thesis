import re
import json
import pprint as p
import sys
from functools import reduce
from itertools import takewhile

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
            if "2d" in results_json and not dimension == "2":
                continue
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
                    if "multi_gpu" in results_json:
                        tmp = re.sub("[^0-9]", "_", results_json).strip("_").split("_")
                        gpus = "".join(list(takewhile(lambda x: x.isdigit(), version)))
                        if gpus not in tmp:
                            continue
                    if "smem_register" in version:
                        continue
                    if v_found is not None:
                        if v_found not in version:
                            continue
                        if "padded" not in v_found and "padded" in version:
                            continue
                    if entry_not_exists(new_db, [dimension, domain_dim, version]):
                        new_db[dimension][domain_dim][version] = {}
                    for stencil_depth, stencil_depth_db in version_db.items():
                        if entry_not_exists(
                            new_db, [dimension, domain_dim, version, stencil_depth]
                        ):
                            new_db[dimension][domain_dim][version][stencil_depth] = {}
                        # if dimension == "3" and int(stencil_depth) > 4:
                        #    continue
                        for iteration, iteration_db in stencil_depth_db.items():
                            if "1024" in results_json and not iteration == "1024":
                                continue
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


# 16 GPU results
# copy("results/results_stencils_heuristic_improved.json", "heid", "heuristic")
copy("results/results_stencils_heuristic_multi_gpu_2_4.json", "heid", "heuristic")
copy("results/results_stencils_heuristic_multi_gpu_8.json", "heid", "heuristic")
copy("results/results_stencils_heuristic_multi_gpu_16.json", "heid", "heuristic")
# print(new_db["3"]["1024"]["16_gpus_base"])

# Two dimensions
copy("results/results_stencils_heuristic_base_2d.json", "heid", "heuristic")
copy("results/results_stencils_heuristic_smem_2d.json", "heid", "heuristic")
copy(
    "results/results_stencils_heuristic_smem_padded_2d.json", "heid", "heuristic"
)

copy("results/results_stencils_autotuned_2d.json", "heid", "autotune")

# copy("results/results_batch_profile_improved_2d.json", "heid", "heuristic")
copy("results/results_batch_profile_base_2d.json", "heid", "heuristic")
copy("results/results_batch_profile_smem_2d.json", "heid", "heuristic")
copy("results/results_batch_profile_smem_padded_2d.json", "heid", "heuristic")

copy("results/results_batch_profile_autotune_2d.json", "heid", "autotune")

copy(
    "results/results_stencils_heuristic_improved_base_1024_iterations.json",
    "heid",
    "heuristic",
)
copy(
    "results/results_stencils_heuristic_improved_smem_1024_iterations.json",
    "heid",
    "heuristic",
)
copy(
    "results/results_stencils_heuristic_improved_smem_padded_1024_iterations.json",
    "heid",
    "heuristic",
)


# 2D IDUN (domain_dim=8192 [512 MiB], 32768 [8 GiB])
copy(
    "results/results_stencils_idun_heuristic_improved_2d.json",
    "idun",
    "heuristic",
)
# 2D IDUN (domain_dim=4096 [128 MiB])
copy(
    "results/results_stencils_idun_heuristic_improved_new_dim_2d.json",
    "idun",
    "heuristic",
)

# 3D IDUN (The others did not have unroll=8 for smem, smem_padded)
copy(
    "results/results_stencils_idun_heuristic_improved_2d_test.json",
    "idun",
    "heuristic",
)

# Three dimensions
copy("results/results_batch_profile_base_3d.json", "heid", "heuristic")
copy("results/results_batch_profile_smem_3d.json", "heid", "heuristic")
copy("results/results_batch_profile_smem_padded_3d.json", "heid", "heuristic")
del new_db["3"]["1024"]["1_gpus_smem_padded_unroll_4"][
    "8"
]  # This fails for heuristic. We should probably regenerate batch_profiles, but the results seems to overlap

copy("results/results_stencils_heuristic_base_3d.json", "heid", "heuristic")
copy("results/results_stencils_heuristic_smem_3d.json", "heid", "heuristic")
copy(
    "results/results_stencils_heuristic_smem_padded_3d.json", "heid", "heuristic"
)


copy("results/results_stencils_autotuned_base_3d.json", "heid", "autotune")
copy("results/results_stencils_autotuned_smem_3d.json", "heid", "autotune")
copy("results/results_stencils_autotuned_smem_padded_3d.json", "heid", "autotune")

copy("results/results_batch_profile_autotune_base_3d.json", "heid", "autotune")
copy("results/results_batch_profile_autotune_smem_3d.json", "heid", "autotune")
copy("results/results_batch_profile_autotune_smem_padded_3d.json", "heid", "autotune")

# 3D IDUN (The others did not have unroll=8 for smem, smem_padded)
# copy(
#    "results/results_stencils_idun_heuristic_improved_3d_test.json",
#    "idun",
#    "heuristic",
# )
#
## 3D IDUN (domain_dim=256 [128 MiB])
# copy(
#    "results/results_stencils_idun_heuristic_improved_new_dim_3d.json",
#    "idun",
#    "heuristic",
# )

# 3D IDUN (domain_dim=256 [128 MiB], 1024 [8 GiB])
copy(
    "results/results_stencils_idun_heuristic_improved_3d.json",
    "idun",
    "heuristic",
)

# 3D HEID (iterations=1024 domain_dim=1024 [8 GiB])
copy(
    "results/results_stencils_heuristic_improved_base_3d_1024_iterations.json",
    "heid",
    "heuristic",
)
copy(
    "results/results_stencils_heuristic_improved_smem_3d_1024_iterations.json",
    "heid",
    "heuristic",
)
copy(
    "results/results_stencils_heuristic_improved_smem_padded_3d_1024_iterations.json",
    "heid",
    "heuristic",
)

with open("results/results_stencils.json", "w") as fp:
    json.dump(new_db, fp)
