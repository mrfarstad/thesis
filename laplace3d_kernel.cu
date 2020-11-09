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

__global__ void GPU_laplace3d(const float* __restrict__ d_u1,
			      float* __restrict__ d_u2)
{
  int   i, j,
        tx, ty,
        bx, by,
        gx, gy,
        blockx, blocky,
        idx, ioff, joff;
  float u2, fourth=1.0f/4.0f, fifth=1.0f/5.0f;

  thread_block tb = this_thread_block();

  //
  // define global indices and array offsets
  //

  i    = threadIdx.x + blockIdx.x*BLOCK_X;
  j    = threadIdx.y + blockIdx.y*BLOCK_Y;

  blockx = BLOCK_X;
  blocky = BLOCK_Y;

  ioff = 1;
  joff = NX;

  idx = i + j*joff;

  if (idx < NX*NY) {
      if (i==0 || i==NX-1 || j==0 || j==NY-1) {
        // Dirichlet b.c.'s
        u2 = d_u1[idx];
      }
      else {
        u2 = (
           //   (d_u1[idx - ioff + joff]
           //+   d_u1[idx + ioff + joff]
           //+   d_u1[idx - ioff - joff]
           //+   d_u1[idx + ioff - joff]) * fifth 
               d_u1[idx - ioff]
           +   d_u1[idx + ioff]
           +   d_u1[idx - joff]
           +   d_u1[idx + joff]) * fourth;
      }
      d_u2[idx] = u2;
  }
}
