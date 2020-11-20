import json as j
import math
import pprint as p
import subprocess

domain_dims = [32, 64, 128, 256, 512, 1024, 2048, 4096]
# NOTE: The 2_gpus output is a copy of 4_gpus. The real output is being generated.
prefix = "results/yme/"
folders = ["1_gpu/1_gpu.txt", "2_gpus/2_gpus.txt", "4_gpus/4_gpus.txt"]

db = {}

for folder in folders:
    gpus = folder.split('/')[0]
    db[gpus] = {}
    ngpus = folder.split('_')[0]
    with open(prefix + folder, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 5):
            tmp = lines[i].split(' ')
            version = tmp[0]
            db[gpus][version] = {}

        #for i in range(0, 6, 5):
        for i in range(0, len(lines), 5):
            tmp = lines[i].split(' ')
            version = tmp[0]

            #domain_dim = int(tmp[1].split("=")[-1].strip())
            domain_dim = tmp[1].split("=")[-1].strip()
            tmp_db = {}
            params = lines[i+1:i+5]

            # Extract block dims for domain dim
            t = [v.strip() for v in params[1].split(',')]
            ttmp = [v.split('=') for v in t]
            blockdims = {}
            for cfg in ttmp:
                block = cfg[0].strip()
                blockdim = cfg[1].strip()
                blockdims[block] = blockdim

            res = subprocess.run(
                    ['./generate_results.sh',
                     ngpus,
                     domain_dim,
                     blockdims['BLOCK_X'],
                     blockdims['BLOCK_Y']],
                    stdout=subprocess.PIPE).stdout.decode('utf-8')
            results = list(filter(None, res.split('\n')))
            blockdims["results"] = [float(result) for result in results]#[res.strip() for res in results]
            db[gpus][version][domain_dim] = blockdims

pretty_db = p.pformat(db)
print(pretty_db)
with open('results.txt', 'w') as out_file:
     out_file.write(pretty_db)

with open('results.json', 'w') as fp:
    j.dump(db, fp)
