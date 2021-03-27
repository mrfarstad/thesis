#include "../include/constants.h"
#include "cooperative_groups.h"
#include "stencils.cu"
#include "stencil_kernel_smem_prefetch.cu"
using namespace cooperative_groups;

__global__ void gpu_stencil_smem_3d(float* __restrict__ d_u1,
			            float* __restrict__ d_u2,
                                    unsigned int kstart,
                                    unsigned int kend)
{
    float u0;
    unsigned int   i, j, k, idx, sidx;
    extern __shared__ float smem[];
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    idx = i + j*NX + k*NX*NY;
    sidx = threadIdx.x + threadIdx.y*(BLOCK_X+SMEM_PAD) + threadIdx.z*(BLOCK_X+SMEM_PAD)*BLOCK_Y;
    if (i<NX && j<NY && k>=kstart && k<=kend) {
        u0 = d_u1[idx];
        smem[sidx] = u0;
    }
    this_thread_block().sync();
    if (check_stencil_border_3d(i, j, k))
        d_u2[idx] = smem_stencil(smem, d_u1, sidx, idx) / STENCIL_COEFF - u0;
}

__global__ void gpu_stencil_smem_2d(float* __restrict__ d_u1,
			            float* __restrict__ d_u2,
                                    unsigned int jstart,
                                    unsigned int jend)
{
    float u0;
    unsigned int   i, j, idx, sidx;
    extern __shared__ float smem[];
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    idx = i + j*NX;
    sidx = threadIdx.x + threadIdx.y*(BLOCK_X+SMEM_PAD);
    if (i<NX && j>=jstart && j<=jend)
    {
        u0 = d_u1[idx];
        smem[sidx] = u0;
    }
    this_thread_block().sync();
    if (check_stencil_border_2d(i, j))
        d_u2[idx] = smem_stencil(smem, d_u1, sidx, idx) / STENCIL_COEFF - u0;
}

__global__ void gpu_stencil_smem_2d_prefetch(float* __restrict__ d_u1,
                                             float* __restrict__ d_u2,
                                             unsigned int jstart,
                                             unsigned int jend)
{
    extern __shared__ float smem[];
    unsigned int i, j, idx, sidx;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    idx = i + j*NX;
    sidx = (threadIdx.x + STENCIL_DEPTH) + (threadIdx.y + STENCIL_DEPTH)*SMEM_P_X;
    prefetch(0, i, j, idx, sidx, smem, d_u1);
    this_thread_block().sync();
    apply_stencil_prefetched(i, j, idx, sidx, smem, d_u2);
}

__global__ void gpu_stencil_smem_2d_unrolled(float* __restrict__ d_u1,
			                     float* __restrict__ d_u2,
                                             unsigned int jstart,
                                             unsigned int jend)
{
    __shared__ float smem[BLOCK_Y][SMEM_X];
    unsigned int i, j, s, idx, sidx, ioff;
    float u;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        idx = (i+ioff) + j*NX;
        if ((i+ioff)<NX && j<NY) smem[threadIdx.y][threadIdx.x+ioff] = d_u1[idx];
    }
    this_thread_block().sync();
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        idx = (i+ioff) + j*NX;
        sidx = (threadIdx.x+ioff)+threadIdx.y*SMEM_X;
        if (check_stencil_border_2d(i+ioff, j))
        {
            u = 0.0f;
            if (s>0)          accumulate_l(&u, *smem, d_u1, sidx, idx, 1, 1);
            else              accumulate_l(&u, *smem, d_u1, sidx, idx, threadIdx.x, 1, 1);
            if (s<UNROLL_X-1) accumulate_r(&u, *smem, d_u1, sidx, idx, 1, 1);
            else              accumulate_r(&u, *smem, d_u1, sidx, idx, BLOCK_X, threadIdx.x, 1, 1);
            accumulate_l(&u, *smem, d_u1, sidx, idx, threadIdx.y, SMEM_X, NX);
            accumulate_r(&u, *smem, d_u1, sidx, idx, BLOCK_Y, threadIdx.y, SMEM_X, NX);
            d_u2[idx] = u / STENCIL_COEFF - smem[threadIdx.y][threadIdx.x+ioff];
        }
    }
}

__global__ void gpu_stencil_smem_2d_unrolled_prefetch(float* __restrict__ d_u1,
                                                      float* __restrict__ d_u2,
                                                      unsigned int jstart,
                                                      unsigned int jend)
{
    extern __shared__ float smem[];
    unsigned int i, j, s, idx, sidx, ioff, si, sj;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    si = threadIdx.x + STENCIL_DEPTH;
    sj = threadIdx.y + STENCIL_DEPTH;
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        idx = (i+ioff) + j*NX;
        sidx = si+ioff + sj*SMEM_P_X;
        prefetch(s, i+ioff, j, idx, sidx, smem, d_u1);
    }
    this_thread_block().sync();
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        idx = (i+ioff) + j*NX;
        sidx = (si+ioff) + sj*SMEM_P_X;
        apply_stencil_prefetched(i+ioff, j, idx, sidx, smem, d_u2);
    }
}

__global__ void gpu_stencil_smem_1d(float* __restrict__ d_u1,
			            float* __restrict__ d_u2,
                                    unsigned int istart,
                                    unsigned int iend)
{
    float u0;
    unsigned int i = threadIdx.x + blockIdx.x*BLOCK_X;
    extern __shared__ float smem[];
    if (i>=istart && i<=iend)
    {
        u0 = d_u1[i];
        smem[threadIdx.x] = u0;
    }
    this_thread_block().sync();
    if (i>=STENCIL_DEPTH && i<NX-STENCIL_DEPTH)
        d_u2[i] = smem_stencil(smem, d_u1, threadIdx.x, i) / STENCIL_COEFF - u0;
}
