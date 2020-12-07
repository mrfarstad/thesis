import json

import pandas as pd

with open("results.json", 'r') as file:
    db = json.loads(file.read())

versions = []
dimensions = []
blockx = []
blocky = []

for gpus, version_db in db.items():
    for version, domain_dim_db in version_db.items():
        for domain_dim, blockdim_db in domain_dim_db.items():
            versions.append(gpus+"_"+version)
            dimensions.append(int(domain_dim))
            blockx.append(blockdim_db['BLOCK_X'])
            blocky.append(blockdim_db['BLOCK_Y'])

df = pd.DataFrame(
        {"version":versions, "dimensions":dimensions, "BLOCK_X":blockx, "BLOCK_Y":blocky},
        columns=["version", "dimensions","BLOCK_X","BLOCK_Y"]
        )

with open('table.txt', 'w') as out_file:
     out = df.to_latex()
     print(out)
     out_file.write(out)
