#ifndef BLOCK_X
#define BLOCK_X 32
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 2
#endif
#ifndef BLOCK_Z
#define BLOCK_Z 2
#endif

#ifndef NGPUS
#define NGPUS 1
#endif

#ifndef ITERATIONS
#define ITERATIONS 64
#endif

#ifndef DIM
#define DIM 256
#endif

#ifndef DIMENSIONS
#define DIMENSIONS 3
#endif

#ifndef STENCIL_DEPTH
#define STENCIL_DEPTH 1
#endif

#ifndef SMEM_PAD
#define SMEM_PAD 0
#endif

#define STENCIL_COEFF ((float)2*DIMENSIONS*STENCIL_DEPTH)

#ifndef UNROLL_X
#define UNROLL_X 4
#endif

#ifndef PREFETCH
#define PREFETCH false
#endif

#ifndef REGISTER
#define REGISTER false
#endif

#define SMEM_X (BLOCK_X*UNROLL_X+SMEM_PAD)
#define SMEM_P_X (SMEM_X+2*STENCIL_DEPTH)
#define SMEM_P_Y (BLOCK_Y+2*STENCIL_DEPTH)
#define REG_SIZE (2*STENCIL_DEPTH+1)

#define NX               (DIM)
#define NY               (DIM)
#define NZ               (DIM)
#if DIMENSIONS==3
#define SIZE             (NX*NY*(unsigned long)NZ)
#elif DIMENSIONS==2
#define SIZE             (NX*(unsigned long)NY)
#else 
#define SIZE             ((unsigned long)NX)
#endif
#define OFFSET           (SIZE/NGPUS)
#define BYTES            (SIZE*sizeof(float))
#define BYTES_PER_GPU    (BYTES/NGPUS)

#define HALO_DEPTH       (STENCIL_DEPTH)
#if DIMENSIONS==3
#define BORDER_SIZE      (NX*NY)
#elif DIMENSIONS==2
#define BORDER_SIZE      (NX)
#else
#define BORDER_SIZE      (1)
#endif
#define GHOST_ZONE       (HALO_DEPTH*BORDER_SIZE)
#define GHOST_ZONE_BYTES (GHOST_ZONE*sizeof(float))
#define HALO_BYTES       (2*GHOST_ZONE_BYTES)

#define INTERNAL_START   (HALO_DEPTH)
#if DIMENSIONS==3
#define INTERNAL_END     (INTERNAL_START+NZ/NGPUS)
#elif DIMENSIONS==2
#define INTERNAL_END     (INTERNAL_START+NY/NGPUS)
#else
#define INTERNAL_END     (INTERNAL_START+NX/NGPUS)
#endif

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

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
