// Optimal blockdims for NX=NY=256 and ITERATIONS=8192 (coop_smem_opt_v1.0)
#ifndef BLOCK_X
#define BLOCK_X 8
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 64
#endif

#define NX 256
#define NY 256

#define ITERATIONS 8192
