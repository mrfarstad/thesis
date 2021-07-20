#include "../include/constants.h"
#include "cooperative_groups.h"
#include "stencils.cu"
#include "prefetch_smem.cu"
#include "stencil_border_check.cu"
using namespace cooperative_groups;

__global__ void smem_3d(float* __restrict__ d_u1,
                        float* __restrict__ d_u2,
                        unsigned int kstart,
                        unsigned int kend)
{
    unsigned int i, j, k, idx, sidx;
    extern __shared__ float smem[];
    i  = threadIdx.x + blockIdx.x*blockDim.x;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
    k  = threadIdx.z + blockIdx.z*blockDim.z;
    idx = i + j*NX + k*NX*NY;
    sidx = threadIdx.x
         + threadIdx.y*blockDim.x*COARSEN_X
         + threadIdx.z*blockDim.x*COARSEN_X*blockDim.y;
    if (check_domain_border_3d(i, j, k, kstart, kend))
        smem[sidx] = d_u1[idx];
    this_thread_block().sync();
    if (check_stencil_border_3d(i, j, k, kstart, kend))
        smem_stencil(smem, d_u1, d_u2, sidx, idx);
}

__global__ void smem_unroll_3d(float* __restrict__ d_u1,
                               float* __restrict__ d_u2,
                               unsigned int kstart,
                               unsigned int kend)
{
    extern __shared__ float smem[];
    unsigned int i, j, k, s, idx, sidx, ioff;
    i  = threadIdx.x + blockIdx.x*blockDim.x*COARSEN_X;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
    k  = threadIdx.z + blockIdx.z*blockDim.z;
#pragma unroll
    for (s=0; s<COARSEN_X; s++) {
        ioff = s*blockDim.x;
        idx = (i+ioff) + j*NX + k*NX*NY;
        sidx = (threadIdx.x+ioff)
             + threadIdx.y*blockDim.x*COARSEN_X
             + threadIdx.z*blockDim.x*COARSEN_X*blockDim.y;
        if (check_domain_border_3d(i+ioff, j, k, kstart, kend))
            smem[sidx] = d_u1[idx];
    }
    this_thread_block().sync();
#pragma unroll
    for (s=0; s<COARSEN_X; s++) {
        ioff = s*blockDim.x;
        idx = (i+ioff) + j*NX + k*NX*NY;
        sidx = (threadIdx.x+ioff)
             + threadIdx.y*blockDim.x*COARSEN_X
             + threadIdx.z*blockDim.x*COARSEN_X*blockDim.y;
        if (check_stencil_border_3d(i+ioff, j, k, kstart, kend))
            smem_unrolled_stencil(d_u1, d_u2, smem, s, idx, sidx);
    }
}

__global__ void smem_2d(float* __restrict__ d_u1,
                        float* __restrict__ d_u2,
                        unsigned int jstart,
                        unsigned int jend)
{
    unsigned int   i, j, idx, sidx;
    extern __shared__ float smem[];
    i  = threadIdx.x + blockIdx.x*blockDim.x;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
    idx = i + j*NX;
    sidx = threadIdx.x + threadIdx.y*blockDim.x*COARSEN_X;
    if (check_domain_border_2d(i, j, jstart, jend))
        smem[sidx] = d_u1[idx];
    this_thread_block().sync();
    if (check_stencil_border_2d(i, j, jstart, jend))
        smem_stencil(smem, d_u1, d_u2, sidx, idx);
}

__global__ void smem_unroll_2d(float* __restrict__ d_u1,
                               float* __restrict__ d_u2,
                               unsigned int jstart,
                               unsigned int jend)
{
    extern __shared__ float smem[];
    unsigned int i, j, s, idx, sidx, ioff;
    i  = threadIdx.x + blockIdx.x*blockDim.x*COARSEN_X;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
#pragma unroll
    for (s=0; s<COARSEN_X; s++) {
        ioff = s*blockDim.x;
        idx = (i+ioff) + j*NX;
        sidx = (threadIdx.x+ioff)+threadIdx.y*blockDim.x*COARSEN_X;
        if (check_domain_border_2d(i+ioff, j, jstart, jend))
            smem[sidx] = d_u1[idx];
    }
    this_thread_block().sync();
#pragma unroll
    for (s=0; s<COARSEN_X; s++) {
        ioff = s*blockDim.x;
        idx = (i+ioff) + j*NX;
        sidx = (threadIdx.x+ioff)+threadIdx.y*blockDim.x*COARSEN_X;
        if (check_stencil_border_2d(i+ioff, j, jstart, jend))
            smem_unrolled_stencil(d_u1, d_u2, smem, s, idx, sidx);
    }
}

__global__ void smem_1d(float* __restrict__ d_u1,
                        float* __restrict__ d_u2,
                        unsigned int istart,
                        unsigned int iend)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    extern __shared__ float smem[];
    if (check_domain_border_1d(i, istart, iend))
        smem[threadIdx.x] = d_u1[i];
    this_thread_block().sync();
    if (check_stencil_border_1d(i, istart, iend))
        smem_stencil(smem, d_u1, d_u2, threadIdx.x, i);
}
