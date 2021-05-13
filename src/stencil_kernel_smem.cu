#include "../include/constants.h"
#include "cooperative_groups.h"
#include "stencils.cu"
#include "prefetch_smem.cu"
#include "stencils_border_check.cu"
using namespace cooperative_groups;

__global__ void smem_3d(float* __restrict__ d_u1,
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
    sidx = threadIdx.x + threadIdx.y*(BLOCK_X+SMEM_PAD) + threadIdx.z*(BLOCK_X+SMEM_PAD)*BLOCK_Y;
    if (check_domain_border_3d(i, j, k, kstart, kend))
        smem[sidx] = d_u1[idx];
    this_thread_block().sync();
    if (check_stencil_border_3d(i, j, k, kstart, kend))
        d_u2[idx] = smem_stencil(smem, d_u1, sidx, idx) / STENCIL_COEFF - smem[sidx];
}

__global__ void smem_2d(float* __restrict__ d_u1,
                        float* __restrict__ d_u2,
                        unsigned int jstart,
                        unsigned int jend)
{
    unsigned int   i, j, idx, sidx;
    extern __shared__ float smem[];
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    idx = i + j*NX;
    sidx = threadIdx.x + threadIdx.y*(BLOCK_X+SMEM_PAD);
    if (check_domain_border_2d(i, j, jstart, jend))
        smem[sidx] = d_u1[idx];
    this_thread_block().sync();
    if (check_stencil_border_2d(i, j, jstart, jend))
        d_u2[idx] = smem_stencil(smem, d_u1, sidx, idx) / STENCIL_COEFF - smem[sidx];
}

__global__ void smem_unroll_2d(float* __restrict__ d_u1,
                               float* __restrict__ d_u2,
                               unsigned int jstart,
                               unsigned int jend)
{
    extern __shared__ float smem[];
    unsigned int i, j, s, idx, sidx, ioff;
    float u;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        idx = (i+ioff) + j*NX;
        sidx = (threadIdx.x+ioff)+threadIdx.y*SMEM_X;
        if (check_domain_border_2d(i+ioff, j, jstart, jend))
            smem[sidx] = d_u1[idx];
    }
    this_thread_block().sync();
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        idx = (i+ioff) + j*NX;
        sidx = (threadIdx.x+ioff)+threadIdx.y*SMEM_X;
        if (check_stencil_border_2d(i+ioff, j, jstart, jend))
        {
            u = 0.0f;
            if (s>0)          accumulate_l(&u, smem, sidx, 1);
            else              accumulate_l(&u, smem, d_u1, sidx, idx, threadIdx.x, 1, 1);
            if (s+1<UNROLL_X) accumulate_r(&u, smem, sidx, 1);
            else              accumulate_r(&u, smem, d_u1, sidx, idx, BLOCK_X, threadIdx.x, 1, 1);
            accumulate_l(&u, smem, d_u1, sidx, idx, threadIdx.y, SMEM_X, NX);
            accumulate_r(&u, smem, d_u1, sidx, idx, BLOCK_Y, threadIdx.y, SMEM_X, NX);
            d_u2[idx] = u / STENCIL_COEFF - smem[sidx];
        }
    }
}

__global__ void smem_1d(float* __restrict__ d_u1,
                        float* __restrict__ d_u2,
                        unsigned int istart,
                        unsigned int iend)
{
    unsigned int i = threadIdx.x + blockIdx.x*BLOCK_X;
    extern __shared__ float smem[];
    if (check_domain_border_1d(i, istart, iend))
        smem[threadIdx.x] = d_u1[i];
    this_thread_block().sync();
    if (check_stencil_border_1d(i, istart, iend))
        d_u2[i] = smem_stencil(smem, d_u1, threadIdx.x, i) / STENCIL_COEFF - smem[threadIdx.x];
}
