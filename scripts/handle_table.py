import json
from math import log

import pandas as pd

with open("results/results_autotune.json", "r") as file:
    db = json.loads(file.read())

versions = []
stencil_depths = []
unrolls = [[] for i in range(4)]

for dimension, dimension_db in db.items():
    domain_dim_db = dimension_db["32768"]
    for version, version_db in domain_dim_db.items():
        for stencil_depth, stencil_depth_db in version_db.items():
            u_idx = 0
            v = version
            if "unroll" in version:
                unroll = int(version[-1])
                u_idx = int(log(unroll) / log(2))
                v = version[: -len("_unroll_x")]
            else:
                versions.append(v)
                stencil_depths.append(stencil_depth)
            unrolls[u_idx].append(
                (stencil_depth_db["BLOCK_X"], stencil_depth_db["BLOCK_Y"])
            )

df = pd.DataFrame(
    {
        "version": versions,
        "stencil_depths": stencil_depths,
        "unroll=1": unrolls[0],
        "unroll=2": unrolls[1],
        "unroll=4": unrolls[2],
        "unroll=8": unrolls[3],
    },
    columns=[
        "version",
        "stencil_depths",
        "unroll=1",
        "unroll=2",
        "unroll=4",
        "unroll=8",
    ],
)

with open("table.txt", "w") as out_file:
    out = df.to_latex(index=False)
    print(out)
    out_file.write(out)
