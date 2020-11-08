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
  int   i, j, k, tx, ty, tz, idx, ioff, joff, koff;
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
  

  ioff = 1;
  joff = NX;
  koff = NX*NY;

  idx = i + j*joff + k*koff;

  __shared__ float smem[BLOCK_Z][BLOCK_Y][BLOCK_X];
  smem[tz][ty][tx] = d_u1[idx];
  __syncthreads();

  if (idx < NX*NY*NZ) {
      if (i==0 || i==NX-1 || j==0 || j==NY-1 || k==0 || k==NZ-1) {
        // Dirichlet b.c.'s
        u2 = d_u1[idx];
      }
      else {
        float tmp = 0.0f;
        if (tx > 0) {
          tmp += smem[tz][ty][tx-1];
        } else {
          tmp += d_u1[idx-ioff];
        }
        if (tx < BLOCK_X-1) {
          tmp += smem[tz][ty][tx+1];
        } else {
          tmp += d_u1[idx+ioff];
        }

        if (ty > 0) {
          tmp += smem[tz][ty-1][tx];
        } else {
          tmp += d_u1[idx-joff];
        }

        if (ty < BLOCK_Y-1) {
          tmp += smem[tz][ty+1][tx];
        } else {
          tmp += d_u1[idx+joff];
        }

        if (tz > 0) {
          tmp += smem[tz-1][ty][tx];
        } else {
          tmp += d_u1[idx-koff];
        }

        if (tz < BLOCK_Z-1) {
          tmp += smem[tz+1][ty][tx];
        } else {
          tmp += d_u1[idx+koff];
        }

        u2 = tmp * sixth;
      }
      d_u2[idx] = u2;
  }
}
