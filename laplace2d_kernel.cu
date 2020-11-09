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

#ifndef NX
#define NX 256
#endif

#ifndef NY
#define NY 256
#endif

#include "cooperative_groups.h"
using namespace cooperative_groups;

__global__ void gpu_laplace2d(const float* __restrict__ d_u1,
			      float* __restrict__ d_u2)
{
  int   i, j, k,
        tx, ty, tz,
        bx, by, bz,
        gx, gy, gz,
        blockx, blocky, blockz,
        idx, sx, sy, sz, ioff, joff, koff;
  float u2, fourth=1.0f/4.0f;

  thread_block tb = this_thread_block();

  //
  // define global indices and array offsets
  //
  //
  i    = threadIdx.x + blockIdx.x*BLOCK_X;
  j    = threadIdx.y + blockIdx.y*BLOCK_Y;

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

  i    = threadIdx.x + blockIdx.x*BLOCK_X;
  j    = threadIdx.y + blockIdx.y*BLOCK_Y;

  ioff = 1;
  joff = NX;

  idx = i + j*joff;

  __shared__ float smem[BLOCK_Y+2][BLOCK_X+2];

  if (tx == 0 && bx != 0)           smem[sy][tx]   = d_u1[idx-ioff];
  if (tx == blockx-1 && bx != gx-1) smem[sy][sx+1] = d_u1[idx+ioff];

  if (ty == 0 && by != 0)           smem[ty][sx]   = d_u1[idx-joff];
  if (ty == blocky-1 && by != gy-1) smem[sy+1][sx] = d_u1[idx+joff];
  
  smem[sy][sx] = d_u1[idx];

  tb.sync();

  if (idx < NX*NY) {
      if (i==0 || i==NX-1 || j==0 || j==NY-1) {
        u2 = d_u1[idx]; // Dirichlet b.c.'s
      }
      else {
        u2 = (smem[sy][sx-1]  +
              smem[sy][sx+1]  +
              smem[sy-1][sx]  +
              smem[sy+1][sx]) * fourth;    
      }
      d_u2[idx] = u2;
  }
}
