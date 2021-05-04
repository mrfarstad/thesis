import json
import pprint as p
import sys
import pandas as pd
from functools import reduce
import statistics


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
versions = [
    ("1_gpus_smem_padded", "1", "heuristic"),
    ("1_gpus_smem_padded", "1", "autotune"),
]

metrics = [db["2"]["32768"][v][d]["8"]["heid"][c] for v, d, c in versions]

for metric_db, version in zip(metrics, versions):  # [v[0] for v in versions]):
    metric_db["version"] = version[0] + " (%s)" % version[2]
    metric_db["time"] = statistics.median(metric_db["time"])
    for k, v in metric_db.items():
        metric_db[k] = [v]

dfs = list(map(pd.DataFrame, metrics))
pd.set_option("display.max_rows", None)
df = pd.concat(dfs)
df.set_index("version", inplace=True)
df = df.reindex(sorted(df.columns), axis=1)
df = df.transpose()
# pd.options.display.width = 0

print(df)
