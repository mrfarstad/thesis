// Optimal blockdims for SX=SY=256 and ITERATIONS=8192 (basic) @ yme
#ifndef BLOCK_X
#define BLOCK_X 32
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 4
#endif

#ifndef NGPUS
#define NGPUS 4
#endif

#ifndef ITERATIONS
#define ITERATIONS 64
#endif

#ifndef DIM
#define DIM 256
#endif

#define NX DIM
#define NY DIM
#define SIZE          (NX*NY)
#define OFFSET        (SIZE/NGPUS)
#define BYTES         (SIZE*sizeof(float))
#define BYTES_PER_GPU (BYTES/NGPUS)
#define HALO_DEPTH    1
#define BORDER_BYTES  (HALO_DEPTH*NX*sizeof(float))
#define HALO_BYTES    (2*BORDER_BYTES)

#ifndef DEBUG
#define DEBUG false
#endif

#ifndef SMEM
#define SMEM false
#endif

#ifndef COOP
#define COOP false
#endif

#define CU checkCudaErrors 
