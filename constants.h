// Optimal blockdims for SX=SY=256 and ITERATIONS=8192 (basic) @ yme
#ifndef BLOCK_X
#define BLOCK_X 32
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 4
#endif

#define NGPUS 2

#ifndef ITERATIONS
#define ITERATIONS 1024
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
#define BYTES_HALO    (2*NX*sizeof(float))

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
#define start_timer cudaEventRecord
