//
// Notes: one thread per node in the 3D block
//

// device code
#include "constants.h"
#include "cooperative_groups.h"
using namespace cooperative_groups;

__global__ void gpu_laplace2d(float* __restrict__ d_u1,
			      float* __restrict__ d_u2)
{
    int   i, j,
          tx, ty,
          idx, ioff, joff;
    float u2 = 0.0f, fourth=1.0f/4.0f;
    
    //
    // define global indices and array offsets
    //
    tx = threadIdx.x;
    ty = threadIdx.y;

    i  = tx + blockIdx.x*BLOCK_X;
    j  = ty + blockIdx.y*BLOCK_Y;

    ioff = 1;
    joff = NX;

    grid_group g = this_grid();
    thread_block tb = this_thread_block();
    __shared__ float smem[BLOCK_Y][BLOCK_X];

    idx = i + j*joff;
    smem[ty][tx] = d_u1[idx];
    tb.sync();
    if (i>=0 && i<=NX-1 && j>=0 && j<=NY-1) {
        if (i==0 || i==NX-1 || j==0 || j==NY-1)
          u2 = d_u1[idx]; // Dirichlet b.c.'s
        else {
              float tmp = 0.0f;
              if (tx > 0)         tmp += smem[ty][tx-1];
              else                tmp += d_u1[idx-ioff];
              if (tx < BLOCK_X-1) tmp += smem[ty][tx+1];
              else                tmp += d_u1[idx+ioff];
              if (ty > 0)         tmp += smem[ty-1][tx];
              else                tmp += d_u1[idx-joff];
              if (ty < BLOCK_Y-1) tmp += smem[ty+1][tx];
              else                tmp += d_u1[idx+joff];
              u2 = tmp * fourth;
        }
        d_u2[idx] = u2;
    }
}
