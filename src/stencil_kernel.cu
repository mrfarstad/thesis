#include "../include/constants.h"
#include "cooperative_groups.h"
#include "stencils.cu"
using namespace cooperative_groups;

__global__ void gpu_stencil_base(float* __restrict__ d_u1,
			           float* __restrict__ d_u2,
                                   unsigned int kstart,
                                   unsigned int kend)
{
    unsigned int   i, j, k, idx;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    idx = i + j*NX + k*NX*NY;
    if (i>=STENCIL_DEPTH && i<NX-STENCIL_DEPTH &&
        j>=STENCIL_DEPTH && j<NY-STENCIL_DEPTH &&
        k>=kstart+STENCIL_DEPTH && k<=kend-STENCIL_DEPTH)
        d_u2[idx] = stencil(d_u1, idx);
}

__global__ void gpu_stencil_smem(float* __restrict__ d_u1,
			           float* __restrict__ d_u2,
                                   unsigned int kstart,
                                   unsigned int kend)
{
    unsigned int   i, j, k, idx, sidx;
    extern __shared__ float smem[];
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    idx = i + j*NX + k*NX*NY;
    sidx = threadIdx.x + threadIdx.y*BLOCK_X + threadIdx.z*BLOCK_X*BLOCK_Y;
    if (i<NX && j<NY && k>=kstart && k<=kend)
        smem[sidx] = d_u1[idx];
    this_thread_block().sync();
    if (i>=STENCIL_DEPTH && i<NX-STENCIL_DEPTH &&
        j>=STENCIL_DEPTH && j<NY-STENCIL_DEPTH &&
        k>=kstart+STENCIL_DEPTH && k<=kend-STENCIL_DEPTH) 
    {
        float u = 0.0f;
        unsigned short d, s;
        bool active;
        for (d=1; d<=STENCIL_DEPTH; d++) {
            active = false;
            for (s=0; s<STENCIL_DEPTH; s++) active |= threadIdx.x == s && i != s;
            if (active) u += d_u1[idx-d];
            else        u += smem[sidx-d];
            active = false;
            for (s=1; s<=STENCIL_DEPTH; s++) active |= threadIdx.x == BLOCK_X-s && i != NX-s;
            if (active) u += d_u1[idx+d];
            else        u += smem[sidx+d];
        }
        for (d=1; d<=STENCIL_DEPTH; d++) {
            active = false;
            for (s=0; s<STENCIL_DEPTH; s++) active |= threadIdx.y == s && j != s;
            if (active) u += d_u1[idx-d*NX];
            else        u += smem[sidx-d*BLOCK_X];
            active = false;
            for (s=1; s<=STENCIL_DEPTH; s++) active |= threadIdx.y == BLOCK_Y-s && j != NY-s;
            if (active) u += d_u1[idx+d*NX];
            else        u += smem[sidx+d*BLOCK_X];
        }
        for (d=1; d<=STENCIL_DEPTH; d++) {
            active = false;
            for (s=0; s<STENCIL_DEPTH; s++) active |= threadIdx.z == s && k != s;
            if (active) u += d_u1[idx-d*NX*NY];
            else        u += smem[sidx-d*BLOCK_X*BLOCK_Y];
            active = false;
            for (s=1; s<=STENCIL_DEPTH; s++) active |= threadIdx.z == BLOCK_Z-s && k != NZ-s;
            if (active) u += d_u1[idx+d*NX*NY];
            else        u += smem[sidx+d*BLOCK_X*BLOCK_Y];
        }
        d_u2[idx] = u / (float) (6 * STENCIL_DEPTH) - smem[sidx];
    }
}

__global__ void gpu_stencil_coop(float* __restrict__ d_u1,
			      float* __restrict__ d_u2)
{
    unsigned int i, j, k, q, x, y, z,
                 xskip, yskip, zskip, 
                 idx;
    float *d_tmp;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    xskip = BLOCK_X * gridDim.x;
    yskip = BLOCK_Y * gridDim.y;
    zskip = BLOCK_Z * gridDim.z;
    grid_group grid = this_grid();
    for (q = 1; q <= ITERATIONS; q++) {
        for (z=k+STENCIL_DEPTH; z<NZ-STENCIL_DEPTH; z+=zskip) {
            for (y=j+STENCIL_DEPTH; y<NY-STENCIL_DEPTH; y+=yskip) {
                for (x=i+STENCIL_DEPTH; x<NX-STENCIL_DEPTH; x+=xskip) {
                    idx = x + y*NX + z*NX*NY;
                    d_u2[idx] = stencil(d_u1, idx);
                }
            }
        }
        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp;
        grid.sync();
    }
}

__global__ void gpu_stencil_coop_smem(float* __restrict__ d_u1,
			      float* __restrict__ d_u2)
{
    int   i, j, k, q, x, y, z, sx, sy, sz, xskip, yskip, zskip, idx;
    float u2, *d_tmp, sixth=1.0f/6.0f;
    sx = threadIdx.x+1;
    sy = threadIdx.y+1;
    sz = threadIdx.z+1;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    xskip = BLOCK_X * gridDim.x;
    yskip = BLOCK_Y * gridDim.y;
    zskip = BLOCK_Z * gridDim.z;
    grid_group grid = this_grid();
    thread_block block = this_thread_block();
    __shared__ float smem[BLOCK_Z+2][BLOCK_Y+2][BLOCK_X+2];
    for (q = 1; q <= ITERATIONS; q++) {
        for (z=k; z<NZ; z+=zskip) {
            for (y=j; y<NY; y+=yskip) {
                for (x=i; x<NX; x+=xskip) {
                    idx = x + y*NX + z*NX*NY;
                    if (threadIdx.x == 0 && x != 0)            smem[sz][sy][sx-1]   = d_u1[idx-1];
                    if (threadIdx.x == BLOCK_X-1 && x != NX-1) smem[sz][sy][sx+1]   = d_u1[idx+1];
                    if (threadIdx.y == 0 && y != 0)            smem[sz][sy-1][sx]   = d_u1[idx-NX];
                    if (threadIdx.y == BLOCK_Y-1 && y != NY-1) smem[sz][sy+1][sx]   = d_u1[idx+NX];
                    if (threadIdx.z == 0 && z != 0)            smem[sz-1][sy][sx]   = d_u1[idx-NX*NY];
                    if (threadIdx.z == BLOCK_Z-1 && z != NZ-1) smem[sz+1][sy][sx]   = d_u1[idx+NX*NY];
                    smem[sz][sy][sx] = d_u1[idx];
                    block.sync();
                    if (x==0 || x==NX-1 || y==0 || y==NY-1 || z==0 || z==NZ-1)
                      u2 = d_u1[idx]; // Dirichlet boundary conditions
                    else {
                      u2 = (smem[sz][sy][sx-1]  +
                            smem[sz][sy][sx+1]  +
                            smem[sz][sy-1][sx]  +
                            smem[sz][sy+1][sx]  +
                            smem[sz-1][sy][sx]  +
                            smem[sz+1][sy][sx]) * sixth;
                    }
                    d_u2[idx] = u2;
                    block.sync();
                }
            }
        }
        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp;
        grid.sync();
    }
}
