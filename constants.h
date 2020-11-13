// Optimal blockdims for SX=SY=256 and ITERATIONS=8192 (basic) @ yme
#ifndef BLOCK_X
#define BLOCK_X 32
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 4
#endif

#define NX 256
#define NY 256

#define ITERATIONS 8192

#ifndef DEBUG
#define DEBUG false
#endif

#ifndef TEST
#define TEST false
#endif

#ifndef SMEM
#define SMEM false
#endif

#ifndef COOP
#define COOP false
#endif
