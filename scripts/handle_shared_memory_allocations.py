import json
from math import log
from pprint import pprint

import pandas as pd


def calculate_smem(kernel, unroll, block_x, block_y):
    sizeof_float = 4
    if kernel == "smem_padded":
        return (
            (block_x * unroll + 2 * stencil_radius)
            * (block_y + 2 * stencil_radius)
            * sizeof_float
        )
    elif kernel == "smem":
        return block_x * unroll * block_y * sizeof_float


def create_alias(kernel, unroll):
    return kernel + "_unroll_" + str(unroll)


kernels = ["smem", "smem_padded"]
unrolls = [1, 2, 4, 8]
stencil_radiuses = [1, 2, 4, 8, 16]
smem_size_db = {
    r: {create_alias(k, u): 0 for k in kernels for u in unrolls}
    for r in stencil_radiuses
}

for kernel in kernels:
    block_x, block_y = 32, 32
    for stencil_radius in stencil_radiuses:
        for k in kernels:
            for u in unrolls:
                smem_size_db[stencil_radius][create_alias(k, u)] = calculate_smem(
                    k, u, block_x, block_y
                )
pprint(smem_size_db)
