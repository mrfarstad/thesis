import json

import pprint as p
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statistics
import sys
import re
import os
from functools import reduce
from numpy import inf
from seaborn_qqplot import pplot
from itertools import dropwhile

opts = [
    "unroll",
    "multi_gpu",
    "multi_gpu_iterations",
    "autotune",
    "version_performance",
    "version_speedup",
    "version_metrics",
    "version_dram",
    "optimal_speedup",
    "iterations",
    "pascal_volta",
    "pascal_coarsened",
    "pascal_coarsened_opt",
    "volta_coarsened_opt",
]

if len(sys.argv) > 1:
    opt = opts[int(sys.argv[1])]
else:
    opt = opts[0]

dimension = "2"
include_smem_register = False
density_plot = False
cherry_pick_multi_gpu = False
median = True
save_fig = True

if not os.path.exists("versions"):
    os.makedirs("versions")

if not os.path.exists("density_plot"):
    os.makedirs("density_plot")


def deep_get(dictionary, keys, default=None):
    return reduce(
        lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
        keys.split("."),
        dictionary,
    )


def is_opt(*args):
    return any([opt == o for o in [*args]])


def in_opt(*args):
    return any([o in opt for o in [*args]])


ylabels = {
    "time": "Performance\n[x]",
    "performance": "Performance\n[1/ms]",
    "arithmetic_intensity": "Arithmetic Intensity",
    "gld_throughput": "Global Memory Load Throughput [GB/s]",
    "shared_load_throughput": "Shared Memory Load Throughput [GB/s]",
    "dram_read_throughput": "Unified Read Throughput",
    "local_throughput": "Local Memory Throughput\n[x]",
    "local_throughput_absolute": "Local Memory Throughput\n[GB/s]",
    "l2_read_throughput": "L2 Read Throughput",
    "l2_write_throughput": "L2 Write Throughput",
    "l2_tex_hit_rate": "l2_tex_hit_rate",
    "local_load_throughput": "Local Load Throughput",
    "local_store_throughput": "Local Store Throughput",
    "local_memory_overhead": "Local Memory Overhead\n[x]",
    "dram_write_throughput": "DRAM Write Throughput\n[x]",
    "dram_write_throughput_absolute": "DRAM Write Throughput\n[GB/s]",
    "dram_utilization": "DRAM Utilization",
    "local_hit_rate": "Register spilling",
    "gld_efficiency": "Global Load Efficiency",
    "gst_efficiency": "Global Store Efficiency",
    "tex_cache_throughput": "Texture Cache Throughput [GB/s]",
    "tex_cache_hit_rate": "Texture Hit Rate",
    "tex_utilization": "Unified Cache Utilization",
    "warp_nonpred_execution_efficiency": "Non-predicated instructions ratio\n(Lower ratio means branch div.)",
    "achieved_occupancy": "Achieved Occupancy",
    "single_precision_fu_utilization": "FP32 Utilization",
    "cf_fu_utilization": "Control Flow FU Utilization Level",
    "issue_slot_utilization": "Issue Slot Utilization Level",
    "ldst_fu_utilization": "Load/Store Unit Utilization Level",
    "tex_fu_utilization": "Tex",
    "stall_memory_dependency": "stall_memory_dependency",
}

with open("results_stencils.json", "r") as f:
    db = json.loads(f.read())

if is_opt("heuristic_occupancy"):
    with open("results_stencils_improved.json", "r") as f:
        heuristic_db = json.loads(f.read())


def entry_exists(nested_list):
    return deep_get(db, ".".join(nested_list)) != None


def getRepeat(v):
    for dimension, dimension_db in db.items():
        for domain_dim, domain_dim_db in dimension_db.items():
            for version, version_db in domain_dim_db.items():
                for stencil_depth, times in version_db.items():
                    return len(times)


def isVersion(v, version):
    if v == "smem" and "prefetch" in version:
        return False
    if not v in version:
        return False
    return True


def getUnrolls(v):
    unrolls = []
    dimension_db = db[dimension]
    for domain_dim, domain_dim_db in dimension_db.items():
        for version, version_db in domain_dim_db.items():
            if not isVersion(v, version):
                continue
            if "unroll" in version:
                unrolls.append(version[-1])
            else:
                unrolls.append("1")
    return list(set(map(int, unrolls)))


def getNGPUS(v):
    ngpus = []
    for dimension, dimension_db in db.items():
        for domain_dim, domain_dim_db in dimension_db.items():
            for version, version_db in domain_dim_db.items():
                if not isVersion(v, version):
                    continue
                ngpus.append(version[0])
    return list(set(map(int, ngpus)))


def createOptimalDataFrame(
    domain_dim,
    config,
    unrolls,
    y,
    metric="time",
    relative_host="heid",
    iterations="8",
    dimension="2",
    host="heid",
):
    dims = []
    depths = []
    versions = []
    ts = []
    vs = ["base", "smem", "smem_padded", "smem_register"]
    if not include_smem_register:
        vs.pop()
    for v in vs:
        v0 = "1_gpus_" + v
        opt = {
            "1": [float(inf)],
            "2": [float(inf)],
            "4": [float(inf)],
            "8": [float(inf)],
            "16": [float(inf)],
        }
        for unroll in unrolls:
            version = v0
            if unroll > 1:
                version += "_unroll_" + str(unroll)
            version_db = deep_get(db, ".".join([dimension, domain_dim, version]))
            for stencil_depth, stencil_depth_db in version_db.items():
                times = stencil_depth_db[iterations][host][config]["time"]
                if not entry_exists(
                    [
                        dimension,
                        domain_dim,
                        version,
                        stencil_depth,
                        iterations,
                        relative_host,
                        config,
                        "time",
                    ]
                ):
                    continue
                if len(times) == 0:
                    continue
                time = statistics.median(times)
                if opt[stencil_depth][0] > time:
                    opt[stencil_depth] = [time, v0]
        for stencil_depth, optimals in opt.items():
            if optimals == [inf]:
                continue
            dims.append(domain_dim)
            depths.append(int(stencil_depth))
            version = optimals[1][len("1_gpus_") :]
            if in_opt("coarsened_opt"):
                versions.append(version + " (coarsened)")
            else:
                versions.append(version + " (optimized)")
            if metric == "performance" or metric == "time":
                if y == "Performance\n[1/ms]":
                    ts.append(1 / optimals[0])
                else:
                    base_time = statistics.median(
                        deep_get(
                            db,
                            ".".join(
                                [
                                    dimension,
                                    domain_dim,
                                    "1_gpus_base",
                                    stencil_depth,
                                    iterations,
                                    relative_host,
                                    "heuristic",
                                    "time",
                                ]
                            ),
                        )
                    )
                    ts.append(base_time / optimals[0])
            else:

                def clean_metric(metric_db, metric):
                    return float(
                        re.sub(
                            "[^0-9.+e]",
                            "",
                            metric_db[metric],
                        )
                    )

                def get_metric(c=config, v=version):
                    v = "1_gpus_" + v
                    return lambda metric: clean_metric(
                        deep_get(
                            db,
                            ".".join(
                                [
                                    dimension,
                                    domain_dim,
                                    v,
                                    stencil_depth,
                                    iterations,
                                    relative_host,
                                    c,
                                ]
                            ),
                        ),
                        metric,
                    )

                def get_metrics(metrics):
                    return sum(list(map(get_metric(), metrics)))

                def get_baseline_metrics(metrics):
                    return sum(list(map(get_metric(v=version), metrics)))

                if metric == "dram_utilization":
                    value = (
                        get_metrics(
                            [
                                "dram_read_throughput",
                                "dram_write_throughput",
                            ]
                        )
                        / 900  # Source: Dissecting the Volta architecture
                    )
                elif metric == "tex_utilization":
                    # Source: Dissecting the Volta architecture
                    value = get_metric()("tex_cache_throughput") / 13800
                elif "fu_utilization" in metric:
                    value = get_metric()(metric) / 10
                elif metric == "local_throughput":
                    value = get_relative_metrics(
                        ["local_load_throughput", "local_store_throughput"]
                    )
                else:
                    value = get_baseline_metrics([metric])
                ts.append(value)
    return pd.DataFrame(
        {
            "domain dimensions (2D)": dims,
            "Stencil Radius": depths,
            "version": versions,
            y: ts,
        },
        columns=["domain dimensions (2D)", "Stencil Radius", "version", y],
    )


def createVersionDataFrame(
    domain_dim,
    config,
    y,
    metric="time",
    host="heid",
    iterations="8",
    db=db,
    relative_host="heid",
    dimension=dimension,
):
    dims = []
    depths = []
    versions = []
    ts = []
    domain_dim_db = deep_get(db, ".".join([dimension, str(domain_dim)]))
    for version, version_db in domain_dim_db.items():
        gpus = version.split("_")[0]
        if int(gpus) > 1 or "unroll" in version:
            continue
        version = version.replace("1_gpus_", "")
        if not include_smem_register and "register" in version:
            continue
        for stencil_depth, stencil_depth_db in version_db.items():
            if iterations not in stencil_depth_db:
                continue
            iteration_db = stencil_depth_db[iterations]
            times = iteration_db[host][config]["time"]
            if len(times) == 0:
                continue
            relative_version = "1_gpus_base"
            base_time = statistics.median(
                deep_get(
                    db,
                    ".".join(
                        [
                            dimension,
                            domain_dim,
                            relative_version,
                            stencil_depth,
                            iterations,
                            relative_host,
                            config,
                            "time",
                        ]
                    ),
                )
            )
            if metric == "performance" or metric == "time":
                if median:
                    dims.append(domain_dim)
                    depths.append(int(stencil_depth))
                    v = version
                    if is_opt("pascal_volta"):
                        v += " (" + ("Volta" if host == "heid" else "Pascal") + ")"
                    versions.append(v)
                    med = statistics.median(times)
                    if y == "Performance\n[1/ms]":
                        ts.append(1 / med)
                    elif y == "Performance\n[x]":
                        ts.append(base_time / med)
                else:
                    for time in times[config]:
                        dims.append(domain_dim)
                        depths.append(int(stencil_depth))
                        versions.append(version)
                        if y == "Performance\n[1/ms]":
                            if density_plot:
                                ts.append(time)
                            else:
                                ts.append(1 / time)
                        elif y == "Performance\n[x]":
                            ts.append(base_time / time)
            else:

                def clean_metric(metric_db, metric):
                    return float(
                        re.sub(
                            "[^0-9.+e]",
                            "",
                            metric_db[metric],
                        )
                    )

                def get_metric(c=config, v=version):
                    v = "1_gpus_" + v
                    return lambda metric: clean_metric(
                        domain_dim_db[v][stencil_depth][iterations][host][c],
                        metric,
                    )

                def get_metrics(metrics):
                    return sum(list(map(get_metric(), metrics)))

                def get_baseline_metrics(metrics):
                    return sum(list(map(get_metric(v=version), metrics)))

                def get_baseline_metrics(metrics):
                    return sum(list(map(get_metric(c="heuristic", v="base"), metrics)))

                def get_relative_metrics(metrics):
                    baseline_metrics = get_baseline_metrics(metrics)
                    metrics = get_metrics(metrics)
                    return metrics / baseline_metrics if baseline_metrics > 0 else 1

                if metric == "dram_utilization":
                    value = (
                        get_metrics(
                            [
                                "dram_read_throughput",
                                "dram_write_throughput",
                            ]
                        )
                        / 900  # Source: Dissecting the Volta architecture
                    )
                elif metric == "tex_utilization":
                    # Source: Dissecting the Volta architecture
                    value = get_metric()("tex_cache_throughput") / 13800
                elif "fu_utilization" in metric:
                    value = get_metric()(metric) / 10
                elif metric == "arithmetic_intensity":
                    value = get_metrics(["flop_count_sp"]) / (
                        get_metrics(
                            ["dram_read_transactions", "dram_write_transactions"]
                        )
                        * 32
                    )
                elif metric == "dram_write_throughput_absolute":
                    value = get_metrics(["dram_write_throughput"])
                elif metric == "local_throughput_absolute":
                    value = get_metrics(
                        [
                            "local_load_throughput",
                            "local_store_throughput",
                        ]
                    )
                else:
                    value = get_baseline_metrics([metric])
                dims.append(domain_dim)
                depths.append(int(stencil_depth))
                versions.append(version)
                ts.append(value)
        if is_opt("arithmetic_intensity"):
            break
    return pd.DataFrame(
        {
            "domain dimensions (2D)": dims,
            "Stencil Radius": depths,
            "version": versions,
            y: ts,
        },
        columns=["domain dimensions (2D)", "Stencil Radius", "version", y],
    )


def createMultiGPUDataFrame(i, domain_dim, config="heuristic", dimension=dimension):
    dims = []
    depths = []
    versions = []
    ts = []
    v = "base"
    domain_dim_db = deep_get(db, ".".join([dimension, domain_dim]))
    for version, version_db in domain_dim_db.items():
        if not isVersion(v, version):
            continue
        gpus = int(version.split("_")[0])
        if "unroll" in version:
            continue
        for stencil_depth, stencil_depth_db in version_db.items():
            iteration_db = stencil_depth_db[i]
            times = iteration_db["heid"][config]["time"]
            single_gpu_time = statistics.median(
                deep_get(
                    db,
                    ".".join(
                        [
                            dimension,
                            domain_dim,
                            "1_gpus_" + v,
                            stencil_depth,
                            i,
                            "heid",
                            config,
                            "time",
                        ]
                    ),
                )
            )
            if median:
                dims.append(domain_dim)
                depths.append(int(stencil_depth))
                versions.append(gpus)
                ts.append(single_gpu_time / statistics.median(times))
            else:
                for time in times:
                    dims.append(domain_dim)
                    depths.append(int(stencil_depth))
                    versions.append(gpus)
                    ts.append(single_gpu_time / time)
    return pd.DataFrame(
        {
            "domain dimensions (2D)": dims,
            "Stencil Radius": depths,
            "version": versions,
            "Performance\n[x]": ts,
        },
        columns=[
            "domain dimensions (2D)",
            "Stencil Radius",
            "version",
            "Performance\n[x]",
        ],
    )


def createUnrollDataFrame(v, domain_dim, unrolls, config, metric, host="heid"):
    dims = []
    depths = []
    versions = []
    ts = []
    iterations = "8"

    domain_dim_db = deep_get(db, ".".join([dimension, domain_dim]))
    v0 = "1_gpus_" + v
    for unroll in unrolls:
        version = v0
        if unroll > 1:
            version += "_unroll_" + str(unroll)
        version_db = deep_get(db, ".".join([dimension, domain_dim, version]))
        for stencil_depth, stencil_depth_db in version_db.items():
            if not entry_exists(
                [
                    dimension,
                    domain_dim,
                    version,
                    stencil_depth,
                    iterations,
                    host,
                    config,
                    metric
                    if not metric == "local_throughput"
                    else "local_load_throughput",
                ]
            ):
                continue
            v1 = "1_gpus_base"
            # Uncomment to compare the isolated effect of autotuning
            # v1 = version
            # if dimension == "3" and "smem_padded" in version and int(stencil_depth) > 4:
            #    continue
            if metric == "time":
                single_unroll_time = statistics.median(
                    deep_get(
                        db,
                        ".".join(
                            [
                                dimension,
                                domain_dim,
                                v1,
                                stencil_depth,
                                iterations,
                                host,
                                "heuristic",
                                metric,
                            ]
                        ),
                    )
                )

                times = stencil_depth_db[iterations][host][config][metric]
                if median:
                    if len(times) == 0:
                        continue
                    med = statistics.median(times)
                    dims.append(domain_dim)
                    depths.append(int(stencil_depth))
                    versions.append(int(unroll))
                    ts.append(single_unroll_time / med)
                else:
                    for time in times:
                        dims.append(domain_dim)
                        depths.append(int(stencil_depth))
                        versions.append(int(unroll))
                        ts.append(single_unroll_time / time)
            else:

                def clean_metric(metric_db, metric):
                    return float(
                        re.sub(
                            "[^0-9.+e]",
                            "",
                            metric_db[metric],
                        )
                    )

                def get_metric(c=config, v=version):
                    return lambda metric: clean_metric(
                        domain_dim_db[v][stencil_depth][iterations][host][c],
                        metric,
                    )

                def get_baseline_metrics(metrics):
                    return sum(list(map(get_metric(v=v1), metrics)))

                def get_heuristic_metrics(metrics):
                    return sum(list(map(get_metric(c="heuristic", v=v1), metrics)))

                def get_metrics(metrics):
                    return sum(list(map(get_metric(), metrics)))

                def get_relative_metrics_autotune(metrics):
                    heuristic = get_heuristic_metrics(metrics)
                    autotune = get_metrics(metrics)
                    return autotune / heuristic if heuristic > 0 else 1

                def get_relative_metrics_unroll(metrics):
                    baseline = get_baseline_metrics(metrics)
                    unrolled = get_metrics(metrics)
                    return unrolled / baseline if baseline > 0 else 1

                def get_relative_metrics(metrics):
                    if opt == "unroll":
                        return get_relative_metrics_unroll(metrics)
                    elif opt == "autotune":
                        return get_relative_metrics_autotune(metrics)

                if metric == "dram_utilization":
                    value = (
                        get_metrics(
                            [
                                "dram_read_throughput",
                                "dram_write_throughput",
                            ]
                        )
                        / 900  # Source: Dissecting the Volta architecture
                    )
                elif metric == "local_throughput":
                    value = get_relative_metrics(
                        [
                            "local_load_throughput",
                            "local_store_throughput",
                        ]
                    )
                elif metric == "dram_write_throughput":
                    value = get_relative_metrics(["dram_write_throughput"])
                else:
                    value = get_relative_metrics([metric])
                dims.append(domain_dim)
                depths.append(int(stencil_depth))
                versions.append(unroll)
                ts.append(value)
        ylabel = ylabels[metric]
    return pd.DataFrame(
        {
            "domain dimensions (2D)": dims,
            "Stencil Radius": depths,
            "version": versions,
            ylabel: ts,
        },
        columns=["domain dimensions (2D)", "Stencil Radius", "version", ylabel],
    )


def formatDomainSize(dim, dimension=dimension):
    dim = int(dim)
    size = dim ** int(dimension) * 4 * 2
    power = 2 ** 10
    n = 0
    power_labels = {0: "", 1: "Ki", 2: "Mi", 3: "Gi"}
    while size > power:
        size /= power
        n += 1
    return str(int(size)) + " " + power_labels[n] + "B"


def createDataFrame(v, d, config, y, i, j):
    dim = "32768" if dimension == "2" else "1024"
    if is_opt("unroll", "autotune"):
        if y == "Performance\n[x]":
            return createUnrollDataFrame(v, dim, getUnrolls(v), config, "time")
        else:
            ylabel = list(ylabels.keys())[list(ylabels.values()).index(y)]
            return createUnrollDataFrame(v, dim, getUnrolls(v), config, ylabel)
    elif in_opt("multi_gpu"):
        if in_opt("_iterations"):
            df = createMultiGPUDataFrame(d, dim)
        else:
            df = createMultiGPUDataFrame("8", d, dimension="2" if i == 0 else "3")
        df = df.sort_values(by=["version", "Stencil Radius"])
        return df
    elif is_opt("version_performance"):
        return createVersionDataFrame(
            d, config, y, "performance", dimension="2" if i == 0 else "3"
        )
    elif is_opt("version_speedup"):
        return createVersionDataFrame(
            d, config, y, "time", dimension="2" if i == 0 else "3"
        )
    elif is_opt("version_dram"):
        metric = list(ylabels.keys())[list(ylabels.values()).index(y)]
        return createVersionDataFrame(
            d,
            config,
            y,
            metric,
            dimension="2" if i == 0 else "3",
        )
    elif in_opt("version", "optimal"):
        metric = list(ylabels.keys())[list(ylabels.values()).index(y)]
        if config == "heuristic":
            return createVersionDataFrame(
                "32768" if i == 0 else "1024",
                config,
                y,
                metric,
                dimension="2" if i == 0 else "3",
            )
        else:
            df = pd.concat(
                [
                    createVersionDataFrame(
                        "32768" if i == 0 else "1024",
                        config,
                        y,
                        metric,
                        dimension="2" if i == 0 else "3",
                    ),
                    createOptimalDataFrame(
                        "32768" if i == 0 else "1024",
                        config,
                        getUnrolls(v),
                        y,
                        metric,
                        dimension="2" if i == 0 else "3",
                    ),
                ],
                ignore_index=True,
            )
            df = df.sort_values(by=["version", "Stencil Radius"])
            return df
    elif is_opt("volta_vs_pascal"):
        return createVersionDataFrame(d, config, y, relative_host="idun")
    elif is_opt("opt_volta_vs_pascal"):
        return createOptimalDataFrame(d, config, getUnrolls(v), y, relative_host="idun")
    elif is_opt("pascal_volta"):
        dimension_idx = 0 if j == 0 else 1
        df = pd.concat(
            [
                createVersionDataFrame(
                    d,
                    config,
                    y,
                    host="idun",
                    relative_host="idun",
                    dimension="2" if i == 0 else "3",
                ),
                createVersionDataFrame(
                    d,
                    config,
                    y,
                    host="heid",
                    relative_host="idun",
                    dimension="2" if i == 0 else "3",
                ),
            ],
            ignore_index=True,
        )
        df = df.sort_values(by=["version", "Stencil Radius"])
        return df
    elif is_opt("pascal_coarsened"):
        return createUnrollDataFrame(
            d, dim, getUnrolls(v), "heuristic", "time", host="idun"
        )
    elif in_opt("coarsened_opt"):
        df = pd.concat(
            [
                createVersionDataFrame(
                    "32768" if i == 0 else "1024",
                    config,
                    y,
                    "time",
                    dimension="2" if i == 0 else "3",
                    host="idun" if in_opt("pascal") else "heid",
                    relative_host="idun" if in_opt("pascal") else "heid",
                ),
                createOptimalDataFrame(
                    "32768" if i == 0 else "1024",
                    config,
                    getUnrolls(v),
                    y,
                    "time",
                    dimension="2" if i == 0 else "3",
                    host="idun" if in_opt("pascal") else "heid",
                    relative_host="idun" if in_opt("pascal") else "heid",
                ),
            ],
            ignore_index=True,
        )
        df = df.sort_values(by=["version", "Stencil Radius"])
        return df
    elif in_opt("pascal"):
        return createVersionDataFrame(d, config, y, host="idun", relative_host="idun")
    elif is_opt("iterations_performance"):
        return createVersionDataFrame(dim, config, y, iterations="1024")
    elif is_opt("iterations"):
        return createVersionDataFrame(dim, config, y, iterations=d)
    elif is_opt("arithmetic_intensity"):
        metric = list(ylabels.keys())[list(ylabels.values()).index(y)]
        return createVersionDataFrame(
            "32768" if i == 0 else "1024",
            config,
            y,
            metric,
            dimension="2" if i == 0 else "3",
        )
    elif is_opt("heuristic_occupancy"):
        metric = list(ylabels.keys())[list(ylabels.values()).index(y)]
        return createVersionDataFrame(dim, config, y, metric, db=d)


def getLegendTitle():
    if is_opt("unroll", "autotune", "pascal_coarsened"):
        return "Coarsening Factor"
    elif in_opt("multi_gpu"):
        return "Number of GPUs"
    elif in_opt("_performance", "coarsened_opt") or is_opt(
        "volta_vs_pascal",
        "pascal_volta",
        "heuristic_occupancy",
        "optimal_speedup",
        "iterations",
    ):
        return "Kernel"


def plotResult():
    dim = "32768" if dimension == "2" else "1024"
    versions = ["base", "smem", "smem_padded", "smem_register"]
    if not include_smem_register:
        versions.pop()
    single_row = in_opt(
        "version",
        "optimal",
        "iterations",
        "multi_gpu",
        "pascal",
        "arithmetic_intensity",
        "heuristic_occupancy",
        "volta",
    )

    if single_row:
        versions = [""]
    if dimension == "3":
        dims = ["32768", "1024"]
    else:
        dims = ["4096", "256"]
    if is_opt("multi_gpu_iterations", "iterations"):
        dims = ["8", "1024"]
    rows = ["(" + chr(96 + i) + ")" for i in range(1, len(versions) + 1)]
    if is_opt("pascal_volta"):
        if dimension == "3":
            dims = ["32768", "1024"]
        else:
            dims = ["4096", "256"]
        cols = ["2D", "3D"]
        versions = ["", ""]
    elif is_opt("multi_gpu_iterations", "iterations"):
        cols = ["{} Iterations".format(col) for col in dims]
    elif is_opt("pascal_coarsened"):
        dims = ["base", "smem", "smem_padded"]
        cols = dims
    elif is_opt("optimal_speedup") or in_opt("coarsened_opt"):
        dims = [dim] * 2
        cols = ["2D", "3D"]
    elif is_opt("version_performance", "version_speedup", "version_dram"):
        dims = ["32768", "1024"]
        cols = ["2D", "3D"]
    elif is_opt("iterations_performance"):
        dims = [dim] * 2
        cols = [""] * 2
    elif is_opt("version", "optimal"):
        dims = [dim] * 2
        cols = [""] * 2
    elif is_opt("unroll", "autotune"):
        cols = versions
        dims = [
            ylabels["time"],
            ylabels[
                "local_throughput" if dimension == "3" else "dram_write_throughput"
            ],
        ]
        rows = rows[: len(dims)]
    elif is_opt("arithmetic_intensity"):
        dims = ["32768", "1024"]
        cols = ["2D", "3D"]
    elif is_opt("heuristic_occupancy"):
        dims = [db, heuristic_db]
        cols = ["Predetermined Heuristic", "Maximized Occupancy Heuristic"]
    else:
        cols = ["2D", "3D"]

    i, j = 0, 0

    if is_opt("iterations_performance"):
        sharey = False
    elif is_opt("version", "optimal", "version_metrics"):
        sharey = False
    else:
        sharey = True

    figx, figy = 6, 5
    if single_row:
        figx = 7
        if is_opt("arithmetic_intensity"):
            figx, figy = 5, 3
        elif is_opt("pascal_volta"):
            if dimension == "3":
                figy = 3.1
            else:
                figy = 2.5
        elif is_opt("pascal_coarsened"):
            if dimension == "3":
                figy = 2.5
            else:
                figy = 3.1
        elif is_opt("pascal_volta"):
            if dimension == "3":
                figy = 3
            else:
                figy = 2.5
        elif is_opt("multi_gpu"):
            if dimension == "3":
                figy = 3
            else:
                figy = 2.5
        elif is_opt("multi_gpu_iterations"):
            if dimension == "2":
                figy = 3.2
            else:
                figy = 2.5
        elif is_opt("iterations"):
            if dimension == "2":
                figy = 3
            else:
                figy = 2.5
        elif is_opt("multi_gpu_iterations"):
            figy = 3
        elif is_opt(
            "version",
            "optimal",
            "iterations",
            "opt_volta_vs_pascal",
            "multi_gpu_iterations",
            "pascal",
            "version_speedup",
            "version_metrics",
            "version_dram",
        ):
            figy = 2.5
        else:
            figy = 3

    if is_opt("unroll", "autotune"):
        figx, figy = 6, 5
    fig, axs = plt.subplots(len(rows), len(cols), sharey=sharey, figsize=(figx, figy))

    pad = 5  # in points

    # https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots

    if single_row:
        a = axs
    else:
        a = axs[0]
    for ax, col in zip(a, cols):
        ax.annotate(
            col,
            xy=(0.5, 1),
            xytext=(0, pad),
            xycoords="axes fraction",
            textcoords="offset points",
            size="medium",
            ha="center",
            va="baseline",
        )

    if not (single_row or is_opt("unroll", "autotune")):
        for ax, row in zip(axs[:, 0], rows):
            ax.annotate(
                row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                size="medium",
                ha="right",
                va="center",
            )

    low = inf
    high = -inf

    y = "Performance\n[1/ms]" if "_performance" in opt else "Performance\n[x]"
    for v in versions:
        for d in dims:
            config = "heuristic"
            if is_opt("autotune") or in_opt("optimal"):
                config = "autotune"
            if single_row:
                ax = axs[i]
            elif is_opt("unroll", "autotune", "pascal_coarsened"):
                ax = axs[i][j]
            else:
                ax = axs[j][i]
            if is_opt("unroll", "autotune"):
                y = dims[i]
            elif is_opt("version_metrics"):
                y = [
                    ylabels["dram_write_throughput_absolute"],
                    ylabels["local_throughput_absolute"],
                ][i]
            elif is_opt("version_dram"):
                y = ylabels["dram_utilization"]
            elif is_opt("iterations_performance"):
                y = [
                    ylabels["performance"],
                    ylabels["time"],
                ][i]
            elif is_opt("version", "optimal"):
                y = [
                    ylabels["dram_utilization"],
                    ylabels["tex_utilization"],
                ][i]
            elif is_opt("arithmetic_intensity"):
                y = ylabels["arithmetic_intensity"]
            elif is_opt("heuristic_occupancy"):
                y = ylabels["performance"]
            df = createDataFrame(v, d, config, y, i, j)
            radiuses = [1, 2, 4, 8, 16]
            df["Stencil Radius"].replace(
                dict(
                    zip(
                        radiuses,
                        ["$\mathregular{R_{" + str(r) + "}}$" for r in radiuses],
                    )
                ),
                inplace=True,
            )
            g = sns.barplot(
                ax=ax,
                x="Stencil Radius",
                y=y,
                hue="version",
                data=df,
                palette="Paired"
                if is_opt("pascal_volta", "optimal_speedup") or in_opt("coarsened_opt")
                else None,
            )
            ys = df[y].tolist()
            min_ys = min(ys)
            max_ys = max(ys)
            if min_ys < low:
                low = min_ys
            if max_ys > high:
                high = max_ys

            def handle_bounds(step, l, h, scale=0.1):
                step = str(step)
                d = step.find(".") + 1
                f = len(step[d:])
                if d == 0:
                    f -= 1
                factor = 10 ** f
                if d > 0:
                    return (
                        math.floor((l - scale * (h - l)) * factor) / factor,
                        math.ceil(h * factor) / factor + scale * (h - l),
                    )
                return (
                    math.floor((l - scale * (h - l)) / factor) * factor,
                    math.ceil(h * factor) / factor + scale * (h - l),
                )

            if in_opt("_performance") and not is_opt("pascal_performance"):
                if i > 0:
                    step = 0.1
                    l, _ = handle_bounds(step, min_ys, max_ys)
                    plt.gca().set_ylim(bottom=l)

            if is_opt("pascal_performance"):
                steps = [0.01, 0.001]
                _, h = handle_bounds(steps[i], min_ys, max_ys)
                ax.set_ylim(0, h)
                ax.set_yticks(np.arange(0, h, step=steps[i]))

            if is_opt(
                "unroll",
                "autotune",
                "iterations",
                "optimal_speedup",
                "version_speedup",
            ):
                steps = [0.1] * 2
                l, h = handle_bounds(steps[i], low, high)
                ax.set_ylim(l, h)
                ax.set_yticks(np.arange(l, h, step=steps[i]))

            if is_opt("version_metrics"):
                steps = [50, 100]
                if i == 0:
                    _, h = handle_bounds(steps[i], low, high, scale=0.2)
                else:
                    _, h = handle_bounds(steps[i], low, high, scale=0.05)
                ax.set_ylim(0, h)
                ax.set_yticks(np.arange(0, h, step=steps[i]))

            if is_opt("pascal_coarsened_opt"):
                steps = [0.5] * 2
                l = 0
                _, h = handle_bounds(steps[i], low, high, scale=0.11)
                ax.set_ylim(l, h)
                ax.set_yticks(np.arange(l, h, step=steps[i]))

            if is_opt("volta_coarsened_opt"):
                steps = [0.1] * 2
                l, h = handle_bounds(steps[i], low, high, scale=0.11)
                ax.set_ylim(l, h)
                ax.set_yticks(np.arange(l, h, step=steps[i]))

            if is_opt("pascal_coarsened"):
                if dimension == "3":
                    steps = [0.1] * 3
                    l, h = handle_bounds(steps[i], low, high)
                else:
                    steps = [0.5] * 3
                    l = 0
                    _, h = handle_bounds(steps[i], low, high)
                ax.set_ylim(l, h)
                ax.set_yticks(np.arange(l, h, step=steps[i]))

            ax.get_legend().remove()
            i = (i + 1) % len(dims)
        j += 1

    if in_opt("multi_gpu"):
        if is_opt("multi_gpu") and dimension == "2":
            scale = 0.1
        elif is_opt("multi_gpu") and dimension == "3":
            scale = 0.059
        elif is_opt("multi_gpu_iterations") and dimension == "3":
            scale = 0.03
        else:
            scale = 0.019
        lscale = scale
        hscale = scale
        if is_opt("multi_gpu_iterations"):
            lscale = 0.05
        l = low - lscale * (high - low)
        h = high + hscale * (high - low)
        ax.set_ylim(l, h)
        if is_opt("multi_gpu") and dimension == "2":
            plt.yticks(np.arange(1, h, step=0.5))
        elif is_opt("multi_gpu_iterations") and dimension == "2":
            plt.yticks(np.arange(1, h, step=1))
        else:
            plt.yticks(np.arange(1, h, step=1))
    elif in_opt("volta_vs_pascal"):
        step = 1
        _, h = handle_bounds(step, low, high)
        plt.ylim(1, h)
        plt.yticks(np.arange(1, h, step=1))
    elif is_opt("pascal_volta"):
        step = 0.5
        _, h = handle_bounds(step, low, high)
        if dimension == "2":
            plt.ylim(0, h - 0.75)
        else:
            plt.ylim(0, h)
        plt.yticks(np.arange(0, h, step=1 if dimension == "3" else 2.5))

    if is_opt("version", "optimal", "version_dram"):
        for i in range(len(axs)):
            axs[i].set_ylim(0, 1)
            axs[i].set_yticks(np.arange(0, 1.1, step=0.1))

    if single_row:
        a = axs[1:].flat
    else:
        a = axs[:, 1:].flat
    if not is_opt("version", "optimal", "version_metrics"):
        plt.setp(a, ylabel="")

    if single_row:
        a = axs[-1]
    else:
        a = axs[-1][-1]
    handles, labels = a.get_legend_handles_labels()
    loc = "lower center"
    if in_opt("_performance", "coarsened_opt", "multi_gpu") or is_opt(
        "unroll",
        "autotune",
        "volta_vs_pascal",
        "pascal_volta",
        "heuristic_occupancy",
        "optimal_speedup",
        "pascal_volta",
        "pascal_coarsened",
        "iterations",
    ):
        loc = "upper center"
    if not is_opt(
        "version",
        "optimal",
        "opt_volta_vs_pascal",
        "pascal",
        "arithmetic_intensity",
        "version_speedup",
        "version_metrics",
        "version_dram",
    ):
        ncol = 5
        if is_opt("pascal_volta", "optimal_speedup") or in_opt("coarsened_opt"):
            ncol = 3
            if include_smem_register:
                ncol = 4
        if not (
            is_opt("pascal_volta", "multi_gpu")
            and dimension == "2"
            or is_opt("pascal_coarsened", "multi_gpu_iterations", "iterations")
            and dimension == "3"
        ):
            fig.legend(
                handles,
                sorted(set(labels))
                if is_opt("pascal_volta", "pascal_coarsened", "optimal")
                else labels,
                loc=loc,
                ncol=ncol,
                title=getLegendTitle(),
            )
    plt.tight_layout()
    if is_opt("unroll", "autotune"):
        plt.subplots_adjust(top=0.84)
    if is_opt("iterations_performance"):
        plt.subplots_adjust(top=0.78)
    elif is_opt("pascal_volta"):
        if dimension == "3":
            plt.subplots_adjust(top=0.67)
    elif is_opt("pascal_coarsened"):
        if dimension == "2":
            plt.subplots_adjust(top=0.75)
    elif is_opt("optimal_speedup") or in_opt("coarsened_opt"):
        plt.subplots_adjust(top=0.67)
    elif is_opt(
        "volta_vs_pascal",
        "pascal_volta",
        "pascal_performance",
        "heuristic_occupancy",
        "version_performance",
    ):
        plt.subplots_adjust(top=0.73)
    elif is_opt("multi_gpu") and dimension == "3":
        plt.subplots_adjust(top=0.73)
    elif is_opt("multi_gpu_iterations") and dimension == "2":
        plt.subplots_adjust(top=0.75)
    elif is_opt("iterations") and dimension == "2":
        plt.subplots_adjust(top=0.73)


"""
    Normalize dataframe entries of the same version and stencil radius parameters.
    This preprocessing stage let's us create a large plot for all measurements in our results,
    which describes the distribution for the whole data set. We use the resulting density plot
    to argue that we should use median over average. Note that x = 0 represents the average.
"""

from itertools import takewhile


def preprocessDataframe(y):
    dims = []
    depths = []
    versions = []
    ts = []
    for dimension, dimension_db in db.items():
        for domain_dim, domain_dim_db in dimension_db.items():
            for version, version_db in domain_dim_db.items():
                if cherry_pick_multi_gpu:
                    if (
                        not int(
                            "".join(list(takewhile(lambda c: c.isdigit(), version)))
                        )
                        > 1
                    ):
                        continue
                else:
                    if (
                        not int(
                            "".join(list(takewhile(lambda c: c.isdigit(), version)))
                        )
                        == 1
                    ):
                        continue
                for stencil_depth, stencil_depth_db in version_db.items():
                    for iteration, iteration_db in stencil_depth_db.items():
                        for host, host_db in iteration_db.items():
                            for conf, times in host_db.items():
                                times = times["time"]
                                if len(times) == 0:
                                    continue
                                performances = times
                                mean = statistics.mean(performances)
                                stdev = statistics.stdev(performances)
                                for time in times:
                                    # Z-score normalization
                                    dims.append(domain_dim)
                                    depths.append(
                                        "$\mathregular{R_{" + stencil_depth + "}}$"
                                    )
                                    versions.append(version)
                                    ts.append((time - mean) / stdev)
    return pd.DataFrame(
        {
            "domain dimensions (2D)": dims,
            "Stencil Radius": depths,
            "version": versions,
            y: ts,
        },
        columns=["domain dimensions (2D)", "Stencil Radius", "version", y],
    )


def plotDensity():
    y = "Time [ms]"
    df = preprocessDataframe(y)
    tmp = sns.displot(df, x=y, kde=True, stat="density")
    # sns.displot(df, x=y, hue="version", kind="ecdf")
    plt.axvline(
        statistics.mean(df[y].tolist()),
        color="b",
        linestyle="dashed",
        linewidth=2,
        label="Mean",
    )
    plt.axvline(
        statistics.median(df[y].tolist()),
        color="r",
        linestyle="dashed",
        linewidth=2,
        label="Median",
    )
    plt.rc("legend", fontsize=14)  # legend fontsize
    plt.tick_params(labelsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.xlabel(y, fontsize=14)
    plt.legend()
    plt.xlim(-2.5, 4)


if density_plot:
    plotDensity()
else:
    plotResult()

if save_fig:
    if density_plot:
        plt.savefig(
            "density_plot/density_plot_"
            + ("multi_gpu" if cherry_pick_multi_gpu else "single_gpu")
            + ".pdf"
        )
    else:
        plt.savefig("versions/" + str(opt) + ".pdf")
else:
    plt.show()
