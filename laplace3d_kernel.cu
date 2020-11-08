//
// Notes: one thread per node in the 3D block
//

// device code
//
#ifndef BLOCK_X
#define BLOCK_X 64
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 2
#endif

#ifndef BLOCK_Z
#define BLOCK_Z 1
#endif


#ifndef NX
#define NX 256
#endif

#ifndef NY
#define NY 256
#endif

#ifndef NZ
#define NZ 256
#endif

#include "cooperative_groups.h"

using namespace cooperative_groups;

__global__ void GPU_laplace3d(const float* __restrict__ d_u1,
			      float* __restrict__ d_u2)
{
  int   i, j, k,
        tx, ty, tz,
        bx, by, bz,
        gx, gy, gz,
        blockx, blocky, blockz,
        idx, sx, sy, sz, ioff, joff, koff;
  float u2, sixth=1.0f/6.0f;

  thread_block tb = this_thread_block();

  //
  // define global indices and array offsets
  //

  i    = threadIdx.x + blockIdx.x*BLOCK_X;
  j    = threadIdx.y + blockIdx.y*BLOCK_Y;
  k    = threadIdx.z + blockIdx.z*BLOCK_Z;

  tx   = threadIdx.x;
  ty   = threadIdx.y;
  tz   = threadIdx.z;

  bx   = blockIdx.x;
  by   = blockIdx.y;
  bz   = blockIdx.z;

  gx   = gridDim.x;
  gy   = gridDim.y;
  gz   = gridDim.z;

  sx = tx+1;
  sy = ty+1;
  sz = tz+1;

  blockx = BLOCK_X;
  blocky = BLOCK_Y;
  blockz = BLOCK_Z;

  ioff = 1;
  joff = NX;
  koff = NX*NY;

  idx = i + j*joff + k*koff;

  __shared__ float smem[BLOCK_Z+2][BLOCK_Y+2][BLOCK_X+2];

  if (tx == 0) {
      if (bx != 0)    smem[sz][sy][tx]   = d_u1[idx-ioff];
      if (bx != gx-1) smem[sz][sy][sx+blockx] = d_u1[idx+blockx];
  }

  if (ty == 0) {
      if (by != 0)    smem[sz][ty][sx]   = d_u1[idx-joff];
      if (by != gy-1) smem[sz][sy+blocky][sx] = d_u1[idx+joff*blocky];
  }

  if (tz == 0) {
      if (bz != 0)    smem[tz][sy][sx]   = d_u1[idx-koff];
      if (bz != gz-1) smem[sz+blockz][sy][sx] = d_u1[idx+koff*blockz];
  }

  smem[sz][sy][sx] = d_u1[idx];

  __syncthreads();

  if (idx < NX*NY*NZ) {
      if (i==0 || i==NX-1 || j==0 || j==NY-1 || k==0 || k==NZ-1) {
        // Dirichlet b.c.'s
        u2 = d_u1[idx];
      }
      else {
        u2 = (smem[sz][sy][sx-1]  +
              smem[sz][sy][sx+1]  +
              smem[sz][sy-1][sx]  +
              smem[sz][sy+1][sx]  +
              smem[sz-1][sy][sx]  +
              smem[sz+1][sy][sx]) * sixth;
      }
      d_u2[idx] = u2;
  }
}
