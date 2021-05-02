import json
import pprint as p
import sys
import pandas as pd
from functools import reduce


def deep_get(dictionary, keys, default=None):
    return reduce(
        lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
        keys.split("."),
        dictionary,
    )


def entry_not_exists(db, nested_list):
    return deep_get(db, ".".join(nested_list)) == None


with open("results/results_stencil_depths.json") as file:
    db = json.loads(file.read())
versions = [("1_gpus_base", "1"), ("1_gpus_base_unroll_2", "1")]
metrics = [db["2"]["32768"][v][d]["8"]["heid"]["heuristic"] for v, d in versions]
for metric_db, version in zip(metrics, [v[0] for v in versions]):
    metric_db["version"] = version
    del metric_db["time"]
    for k, v in metric_db.items():
        metric_db[k] = [v]
dfs = list(map(pd.DataFrame, metrics))
pd.set_option("display.max_rows", None)
dfs = pd.concat(dfs)
dfs.set_index("version", inplace=True)
dfs = dfs.transpose()

print(dfs)
