#include "../include/constants.h"
#include "cooperative_groups.h"
using namespace cooperative_groups;

__global__ void gpu_laplace3d_base(float* __restrict__ d_u1,
			           float* __restrict__ d_u2,
                                   int kstart,
                                   int kend)
{
    int   i, j, k, idx;
    float u2 = 0.0f, sixth=1.0f/6.0f;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    idx = i + j*NX + k*NX*NY;
    if (i<NX && j<NY && k>=kstart && k<=kend) {
        if (i==0 || i==NX-1 || j==0 || j==NY-1 || k==kstart || k==kend)
          u2 = d_u1[idx]; // Dirichlet boundary conditions
        else {
          u2 = (d_u1[idx-1]      +
                d_u1[idx+1]      +
                d_u1[idx-NX]     +
                d_u1[idx+NX]     +
                d_u1[idx-NX*NY]  +
                d_u1[idx+NX*NY]) * sixth;
        }
        d_u2[idx] = u2;
    }
}

__global__ void gpu_laplace3d_smem(float* __restrict__ d_u1,
			           float* __restrict__ d_u2,
                                   int kstart,
                                   int kend)
{
    int   i, j, k, sx, sy, sz, idx;
    float u2 = 0.0f, sixth=1.0f/6.0f;
    sx = threadIdx.x+1;
    sy = threadIdx.y+1;
    sz = threadIdx.z+1;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    thread_block tb = this_thread_block();
    __shared__ float smem[BLOCK_Z+2][BLOCK_Y+2][BLOCK_X+2];
    idx = i + j*NX + k*NX*NY;
    bool active = i<NX && j<NY && k>=kstart && k<=kend;
    if (active) {
        if (threadIdx.x == 0 && i != 0)            smem[sz][sy][sx-1]   = d_u1[idx-1];
        if (threadIdx.x == BLOCK_X-1 && i != NX-1) smem[sz][sy][sx+1]   = d_u1[idx+1];
        if (threadIdx.y == 0 && j != 0)            smem[sz][sy-1][sx]   = d_u1[idx-NX];
        if (threadIdx.y == BLOCK_Y-1 && j != NY-1) smem[sz][sy+1][sx]   = d_u1[idx+NX];
        if (threadIdx.z == 0 && k != kstart)       smem[sz-1][sy][sx]   = d_u1[idx-NX*NY];
        if (threadIdx.z == BLOCK_Z-1 && k != kend) smem[sz+1][sy][sx]   = d_u1[idx+NX*NY];
        smem[sz][sy][sx] = d_u1[idx];
    }
    tb.sync();
    if (active) {
        if (i==0 || i==NX-1 || j==0 || j==NY-1 || k==kstart || k==kend)
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
    }
}

__global__ void gpu_laplace3d_coop(float* __restrict__ d_u1,
			      float* __restrict__ d_u2)
{
    int   i, j, k, q, x, y, z,
          xskip, yskip, zskip, 
          idx;
    float u2, *d_tmp, sixth=1.0f/6.0f;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    xskip = BLOCK_X * gridDim.x;
    yskip = BLOCK_Y * gridDim.y;
    zskip = BLOCK_Z * gridDim.z;
    grid_group grid = this_grid();
    for (q = 1; q <= ITERATIONS; q++) {
        for (z=k; z<NZ; z+=zskip) {
            for (y=j; y<NY; y+=yskip) {
                for (x=i; x<NX; x+=xskip) {
                    idx = x + y*NX + z*NX*NY;
                    if (x==0 || x==NX-1 || y==0 || y==NY-1 || z==0 || z==NZ-1)
                      u2 = d_u1[idx]; // Dirichlet boundary conditions
                    else {
                      u2 = (d_u1[idx-1]      +
                            d_u1[idx+1]      +
                            d_u1[idx-NX]     +
                            d_u1[idx+NX]     +
                            d_u1[idx-NX*NY]  +
                            d_u1[idx+NX*NY]) * sixth;    
                    }
                    d_u2[idx] = u2;
                }
            }
        }
        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp;
        grid.sync();
    }
}

__global__ void gpu_laplace3d_coop_smem(float* __restrict__ d_u1,
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
