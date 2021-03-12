#include "../include/constants.h"
#include "cooperative_groups.h"
#include "stencils.cu"
using namespace cooperative_groups;

typedef bool (*check_t) (unsigned int,unsigned int,unsigned int);

__device__ inline bool is_internal(unsigned int k, unsigned int kstart, unsigned int kend) {
#if NGPUS==1
    return k>=kstart+STENCIL_DEPTH && k<=kend-STENCIL_DEPTH;
#else
    return k>=kstart+2*STENCIL_DEPTH && k<=kend-2*STENCIL_DEPTH;
#endif
}

__device__ inline bool is_ghost_zone(unsigned int k, unsigned int kstart, unsigned int kend) {
    return (k>=kstart+STENCIL_DEPTH && k<=kstart+2*STENCIL_DEPTH) ||
           (k>=kend-2*STENCIL_DEPTH && k<=kend-STENCIL_DEPTH);
}

__device__ void gpu_stencil_base_kernel(float* __restrict__ d_u1,
                                 float* __restrict__ d_u2,
                                 unsigned int kstart,
                                 unsigned int kend,
                                 check_t check)
{
    unsigned int   i, j, k, idx;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    idx = i + j*NX + k*NX*NY;
    if (i>=STENCIL_DEPTH && i<NX-STENCIL_DEPTH &&
        j>=STENCIL_DEPTH && j<NY-STENCIL_DEPTH &&
        check(k, kstart, kend))
        d_u2[idx] = stencil(d_u1, idx);
}

__global__ void gpu_stencil_base(float* __restrict__ d_u1,
                                 float* __restrict__ d_u2,
                                 unsigned int kstart,
                                 unsigned int kend)
{
    gpu_stencil_base_kernel(d_u1, d_u2, kstart, kend, is_internal);
}

__global__ void gpu_stencil_base_ghost_zone(float* __restrict__ d_u1,
                                            float* __restrict__ d_u2,
                                            unsigned int kstart,
                                            unsigned int kend)
{
    gpu_stencil_base_kernel(d_u1, d_u2, kstart, kend, is_ghost_zone);
}


__device__ void gpu_stencil_smem_kernel(float* __restrict__ d_u1,
			                float* __restrict__ d_u2,
                                        unsigned int kstart,
                                        unsigned int kend,
                                        check_t check)
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
        check(k, kstart, kend))
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

__global__ void gpu_stencil_smem(float* __restrict__ d_u1,
                                 float* __restrict__ d_u2,
                                 unsigned int kstart,
                                 unsigned int kend)
{
    gpu_stencil_smem_kernel(d_u1, d_u2, kstart, kend, is_internal);
}

__global__ void gpu_stencil_smem_ghost_zone(float* __restrict__ d_u1,
                                            float* __restrict__ d_u2,
                                            unsigned int kstart,
                                            unsigned int kend)
{
    gpu_stencil_smem_kernel(d_u1, d_u2, kstart, kend, is_ghost_zone);
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
