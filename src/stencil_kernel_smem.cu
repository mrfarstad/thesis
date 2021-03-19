#include "../include/constants.h"
#include "cooperative_groups.h"
using namespace cooperative_groups;

__device__ float smem_stencil(float* smem, float* d_u1, unsigned int sidx, unsigned int idx, float u) {
    unsigned int d;
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
    return u;
}

__global__ void gpu_stencil_smem_3d(float* __restrict__ d_u1,
			            float* __restrict__ d_u2,
                                    unsigned int kstart,
                                    unsigned int kend)
{
    float u = 0.0f, u0;
    unsigned int   i, j, k, idx, sidx;
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
        d_u2[idx] = smem_stencil(smem, d_u1, sidx, idx, u) / (float) (6 * STENCIL_DEPTH) - u0;
    }
}

__global__ void gpu_stencil_smem_2d(float* __restrict__ d_u1,
			            float* __restrict__ d_u2,
                                    unsigned int kstart,
                                    unsigned int kend)
{
    float u = 0.0f, u0;
    unsigned int   i, j, k, idx, sidx;
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
        d_u2[idx] = smem_stencil(smem, d_u1, sidx, idx, u) / (float) (6 * STENCIL_DEPTH) - u0;
    }
}

__global__ void gpu_stencil_smem_1d(float* __restrict__ d_u1,
			            float* __restrict__ d_u2,
                                    unsigned int kstart,
                                    unsigned int kend)
{
    float u = 0.0f, u0;
    unsigned int   i, j, k, idx, sidx;
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
        d_u2[idx] = smem_stencil(smem, d_u1, sidx, idx, u) / (float) (6 * STENCIL_DEPTH) - u0;
    }
}
