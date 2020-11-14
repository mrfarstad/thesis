// Optimal blockdims for SX=SY=256 and ITERATIONS=8192 (basic) @ yme
#ifndef BLOCK_X
#define BLOCK_X 32
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 4
#endif

#define NGPUS 4

#define NX 256
#define NY 256
#define SIZE          (NX*NY)
#define OFFSET        (SIZE/NGPUS)
#define BYTES         (SIZE*sizeof(float))
#define BYTES_PER_GPU (BYTES/NGPUS)
#define BYTES_BORDER  (2*NX*sizeof(float))

#define ITERATIONS 8192
//#define ITERATIONS 1

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

#define CU checkCudaErrors 
#define start_timer cudaEventRecord
