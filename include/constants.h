#ifndef HEURISTIC
#define HEURISTIC false
#endif

#ifndef ARCH
#define ARCH volta
#endif

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

#ifndef RADIUS
#define RADIUS 1
#endif

#ifndef SMEM_PAD
#define SMEM_PAD 0
#endif

#define STENCIL_COEFF ((float)2*DIMENSIONS*RADIUS)

#ifndef COARSEN_X
#define COARSEN_X 4
#endif

#ifndef PADDED
#define PADDED false
#endif

#ifndef REGISTER
#define REGISTER false
#endif

#define VOLTA_SMEM 98304
#define PASCAL_SMEM 49152

#define SMEM_X (BLOCK_X*COARSEN_X+SMEM_PAD)
#define SMEM_P_X (SMEM_X+2*RADIUS)
#define SMEM_P_Y (BLOCK_Y+2*RADIUS)
#define SMEM_P_Z (BLOCK_Z+2*RADIUS)
#define REG_SIZE (2*RADIUS+1)

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

#define GHOST_ZONE_DEPTH (RADIUS)
#if DIMENSIONS==3
#define BORDER_SIZE      (NX*NY)
#elif DIMENSIONS==2
#define BORDER_SIZE      (NX)
#else
#define BORDER_SIZE      (1)
#endif
#define GHOST_ZONE       (RADIUS*BORDER_SIZE)
#define GHOST_ZONE_BYTES (GHOST_ZONE*sizeof(float))

#define INTERNAL_START   (GHOST_ZONE_DEPTH)
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
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
