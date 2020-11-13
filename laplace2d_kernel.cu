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
          tx, ty, sx, sy,
          idx, ioff, joff;
    float u2 = 0.0f, fourth=1.0f/4.0f;
    
    //
    // define global indices and array offsets
    //
    tx = threadIdx.x;
    ty = threadIdx.y;

    sx = tx+1;
    sy = ty+1;

    i  = tx + blockIdx.x*BLOCK_X;
    j  = ty + blockIdx.y*BLOCK_Y;

    ioff = 1;
    joff = NX;

    grid_group g = this_grid();
    thread_block tb = this_thread_block();
    __shared__ float smem[BLOCK_Y+2][BLOCK_X+2];

    idx = i + j*joff;
    if (i != 0)           smem[sy][sx-1]   = d_u1[idx-ioff];
    if (i != NX-1)        smem[sy][sx+1]   = d_u1[idx+ioff];
    if (j != 0)           smem[sy-1][sx]   = d_u1[idx-joff];
    if (j != NY-1)        smem[sy+1][sx]   = d_u1[idx+joff];
    smem[sy][sx] = d_u1[idx];
    tb.sync();
    if (i>=0 && i<=NX-1 && j>=0 && j<=NY-1) {
        if (i==0 || i==NX-1 || j==0 || j==NY-1)
          u2 = d_u1[idx]; // Dirichlet b.c.'s
        else {
          u2 = (smem[sy][sx-1]  +
                smem[sy][sx+1]  +
                smem[sy-1][sx]  +
                smem[sy+1][sx]) * fourth;
        }
        d_u2[idx] = u2;
    }
}
