import math
import pprint as p

domain_dims = [32, 64, 128, 256, 512, 1024, 2048, 4096]
# NOTE: The 2_gpus output is a copy of 4_gpus. The real output is being generated.
folders = ["2_gpus/2_gpus.txt", "4_gpus/4_gpus.txt"]

db = {}

for folder in folders:
    gpus = folder.split('/')[0]
    db[gpus] = {}
    with open(folder, "r") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 5):
            tmp = lines[i].split(' ')
            version = tmp[0]
            db[gpus][version] = {}

        for i in range(0, len(lines), 5):
            tmp = lines[i].split(' ')
            version = tmp[0]

            domain_dim = tmp[1].split("=")[-1].strip()
            tmp_db = {}
            params = lines[i+1:i+5]

            # Extract block dims for domain dim
            t = [v.strip() for v in params[1].split(',')]
            ttmp = [v.split('=') for v in t]
            blockdims = {}
            for cfg in ttmp:
                block = cfg[0].strip()
                blockdim = int(cfg[1].strip())
                blockdims[block] = blockdim

            db[gpus][version][domain_dim] = blockdims



p.pprint(db)
