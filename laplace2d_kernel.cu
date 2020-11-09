//
// Notes: one thread per node in the 3D block
//

// device code

#ifndef BLOCK_X
#define BLOCK_X 32
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 32
#endif

#include "cooperative_groups.h"
using namespace cooperative_groups;

__global__ void gpu_laplace2d(float* __restrict__ d_u1,
			      float* __restrict__ d_u2,
                              int nx,
                              int ny,
                              int iter)
{
    int   i, j, q,
          tx, ty, sx, sy,
          xskip, yskip, 
          idx, ioff, joff;
    float u2, *d_tmp, fourth=1.0f/4.0f;
    
    //
    // define global indices and array offsets
    //
    tx = threadIdx.x;
    ty = threadIdx.y;

    sx = tx+1;
    sy = ty+1;

    i  = tx + blockIdx.x*BLOCK_X;
    j  = ty + blockIdx.y*BLOCK_Y;

    xskip = BLOCK_X * gridDim.x;
    yskip = BLOCK_Y * gridDim.y;
    
    ioff = 1;
    joff = nx;
    grid_group g = this_grid();
    thread_block tb = this_thread_block();

    __shared__ float smem[BLOCK_Y][BLOCK_X];

    for (q = 1; q <= iter; q++) {
        for (int y=j; y<ny; y+=yskip) {
            for (int x=i; x<nx; x+=xskip) {
                idx = x + y*joff;

                tb.sync();
                smem[ty][tx] = d_u1[idx];
                tb.sync();

                if (idx < nx*ny) {
                    if (x==0 || x==nx-1 || y==0 || y==ny-1) {
                      u2 = d_u1[idx]; // Dirichlet b.c.'s
                    }
                    else {
                      float tmp = 0.0f;
                      if (tx > 0) {
                        tmp += smem[ty][tx-1];
                      } else {
                          tmp += d_u1[idx-ioff];
                      }
                      if (tx < BLOCK_X-1) {
                        tmp += smem[ty][tx+1];
                      } else {
                        tmp += d_u1[idx+ioff];
                      }

                      if (ty > 0) {
                        tmp += smem[ty-1][tx];
                      } else {
                        tmp += d_u1[idx-joff];
                      }
                      if (ty < BLOCK_Y-1) {
                        tmp += smem[ty+1][tx];
                      } else {
                        tmp += d_u1[idx+joff];
                      }
                      u2 = tmp * fourth;
                    }
                    d_u2[idx] = u2;
                }
            }
        }
        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp; // swap d_u1 and d_u2
        g.sync();
    }
}
