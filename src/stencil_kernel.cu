#include "../include/constants.h"
#include "cooperative_groups.h"
using namespace cooperative_groups;

__global__ void gpu_stencil_smem(float* __restrict__ d_u1,
			           float* __restrict__ d_u2,
                                   unsigned int kstart,
                                   unsigned int kend)
{
    float u = 0.0f, u0;
    unsigned int   i, j, k, idx, sidx, d;
    extern __shared__ float smem[];
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    idx = i + j*NX + k*NX*NY;
    sidx = threadIdx.x + threadIdx.y*BLOCK_X + threadIdx.z*BLOCK_X*BLOCK_Y;
    if (i<NX && j<NY && k>=kstart && k<=kend) {
        u0 = d_u1[idx];
        smem[sidx] = u0;
    }
    this_thread_block().sync();
    if (i>=STENCIL_DEPTH && i<NX-STENCIL_DEPTH &&
        j>=STENCIL_DEPTH && j<NY-STENCIL_DEPTH &&
        k>=kstart+STENCIL_DEPTH && k<=kend-STENCIL_DEPTH) 
    {
#pragma unroll
        for (d=STENCIL_DEPTH; d>=1; d--) {
            if (threadIdx.x >= d)        u += smem[sidx-d];
            else                         u += d_u1[idx-d];
        }
#pragma unroll
        for (d=1; d<=STENCIL_DEPTH; d++) {
            if (threadIdx.x+d < BLOCK_X) u += smem[sidx+d];
            else                         u += d_u1[idx+d];
        }
#pragma unroll
        for (d=STENCIL_DEPTH; d>=1; d--) {
            if (threadIdx.y >= d)        u += smem[sidx-d*BLOCK_X];
            else                         u += d_u1[idx-d*NX];
        }
#pragma unroll
        for (d=1; d<=STENCIL_DEPTH; d++) {
            if (threadIdx.y+d < BLOCK_Y) u += smem[sidx+d*BLOCK_X];
            else                         u += d_u1[idx+d*NX];
        }
#pragma unroll
        for (d=STENCIL_DEPTH; d>=1; d--) {
            if (threadIdx.z >= d)        u += smem[sidx-d*BLOCK_X*BLOCK_Y];
            else                         u += d_u1[idx-d*NX*NY];
        }
#pragma unroll
        for (d=1; d<=STENCIL_DEPTH; d++) {
            if (threadIdx.z+d < BLOCK_Z) u += smem[sidx+d*BLOCK_X*BLOCK_Y];
            else                         u += d_u1[idx+d*NX*NY];
        }
        d_u2[idx] = u / (float) (6 * STENCIL_DEPTH) - u0;
    }
}

__global__ void gpu_stencil_coop(float* __restrict__ d_u1,
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
