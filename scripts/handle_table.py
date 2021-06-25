import json
from math import log

import pandas as pd

with open("results/results_autotune.json", "r") as file:
    db = json.loads(file.read())

versions = []
stencil_depths = []
unrolls = [[] for i in range(4)]

dimension = "2"
dimension_db = db[dimension]
domain_dim_db = dimension_db["1024" if dimension == "3" else "32768"]
for version, version_db in domain_dim_db.items():
    for stencil_depth in ["1", "2", "4", "8", "16"]:
        u_idx = 0
        v = version
        if "unroll" in version:
            unroll = int(version[-1])
            u_idx = int(log(unroll) / log(2))
        else:
            versions.append(v)
            stencil_depths.append(stencil_depth)
        if stencil_depth in version_db:
            stencil_depth_db = version_db[stencil_depth]
            unrolls[u_idx].append(
                (
                    stencil_depth_db["BLOCK_X"],
                    stencil_depth_db["BLOCK_Y"],
                    stencil_depth_db["BLOCK_Z"],
                )
                if dimension == "3"
                else (
                    stencil_depth_db["BLOCK_X"],
                    stencil_depth_db["BLOCK_Y"],
                )
            )
        else:
            unrolls[u_idx].append("N/A")

df = pd.DataFrame(
    {
        "Kernel": versions,
        "Radius": stencil_depths,
        "CF=1": unrolls[0],
        "CF=2": unrolls[1],
        "CF=4": unrolls[2],
        "CF=8": unrolls[3],
    },
    columns=[
        "Kernel",
        "Radius",
        "CF=1",
        "CF=2",
        "CF=4",
        "CF=8",
    ],
)

with open("table_" + str(dimension) + "d.txt", "w") as out_file:
    out = df.to_latex(index=False)
    print(out)
    out_file.write(out)
