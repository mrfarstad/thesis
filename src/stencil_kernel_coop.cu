#include "../include/constants.h"
#include "cooperative_groups.h"
using namespace cooperative_groups;

__global__ void coop(float* __restrict__ d_u1,
                     float* __restrict__ d_u2)
{
    unsigned int i, j, k, q, x, y, z,
                 xskip, yskip, zskip, 
                 idx;
    unsigned int s;
    float *d_tmp, u, u0;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    xskip = BLOCK_X * gridDim.x;
    yskip = BLOCK_Y * gridDim.y;
    zskip = BLOCK_Z * gridDim.z;
    grid_group grid = this_grid();
    // TODO: I believe this is inefficient as each thread skips so many positions when handling multiple elements..
    // I would argue against this, and try to let each thread handle a consecutive block.
    // The version we have from before is obviously simpler, but its simplicitly impacts the performance negatively.
    for (q = 1; q <= ITERATIONS; q++) {
        for (z=k+STENCIL_DEPTH; z<NZ-STENCIL_DEPTH; z+=zskip) {
            for (y=j+STENCIL_DEPTH; y<NY-STENCIL_DEPTH; y+=yskip) {
                for (x=i+STENCIL_DEPTH; x<NX-STENCIL_DEPTH; x+=xskip) {
                    idx = x + y*NX + z*NX*NY;
                    u = 0.0f;
                    u0 = d_u1[idx];
                    for (s=STENCIL_DEPTH; s>=1; s--)
                        u+=d_u1[idx-s];
                    for (s=1; s<=STENCIL_DEPTH; s++)
                        u+=d_u1[idx+s];
                    for (s=STENCIL_DEPTH; s>=1; s--)
                        u+=d_u1[idx-s*NX];
                    for (s=1; s<=STENCIL_DEPTH; s++)
                        u+=d_u1[idx+s*NX];
                    for (s=STENCIL_DEPTH; s>=1; s--)
                        u+=d_u1[idx-s*NX*NY];
                    for (s=1; s<=STENCIL_DEPTH; s++)
                        u+=d_u1[idx+s*NX*NY];
                    d_u2[idx] = u / (float) (6 * STENCIL_DEPTH) - u0;
                }
            }
        }
        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp;
        grid.sync();
    }
}
