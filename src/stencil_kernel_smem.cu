#include "../include/constants.h"
#include "cooperative_groups.h"
#include "stencils.cu"
#include "stencil_kernel_smem_padded.cu"
#include "stencils_border_check.cu"
using namespace cooperative_groups;

__global__ void gpu_stencil_smem_3d(float* __restrict__ d_u1,
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

__global__ void gpu_stencil_smem_2d(float* __restrict__ d_u1,
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

__global__ void gpu_stencil_smem_2d_register(float* __restrict__ d_u1,
                                             float* __restrict__ d_u2,
                                             unsigned int jstart,
                                             unsigned int jend)
{
    unsigned int i, j, idx, sidx;
    extern __shared__ float smem[];
    float yval[2*STENCIL_DEPTH+1];
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    idx = i + j*NX;
    sidx = (threadIdx.x + STENCIL_DEPTH) + threadIdx.y*SMEM_P_X;
    if (check_domain_border_2d(i, j, jstart, jend))
    {
        prefetch_i_left(i, sidx, idx, smem, d_u1);
        prefetch_i_right(i, sidx, idx, smem, d_u1);
        prefetch_reg_j_down(j, idx, yval, d_u1);
        prefetch_reg_j_up(j, idx, yval, d_u1);
        yval[STENCIL_DEPTH] = d_u1[idx];
        smem[sidx] = yval[STENCIL_DEPTH];
    }
    this_thread_block().sync();
    if (check_stencil_border_2d(i, j, jstart, jend))
    {
        d_u2[idx] = smem_reg_stencil(smem, yval, sidx) / STENCIL_COEFF - yval[STENCIL_DEPTH];
    }
}

__global__ void gpu_stencil_smem_2d_unrolled_register(float* __restrict__ d_u1,
                                                      float* __restrict__ d_u2,
                                                      unsigned int jstart,
                                                      unsigned int jend)
{
    unsigned int i, j, s, si, i_off, idx, sidx, ioff;
    extern __shared__ float smem[];
    float yval[UNROLL_X][REG_SIZE], *ys;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    idx = i + j*NX;
    si = threadIdx.x + STENCIL_DEPTH;
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        i_off = i + ioff;
        idx = i_off + j*NX;
        sidx = (si+ioff) + threadIdx.y*SMEM_P_X;
        ys = yval[s];
        if (check_domain_border_2d(i_off, j, jstart, jend))
        {
            if (s==0)          prefetch_i_left(i_off, sidx, idx, smem, d_u1);
            if (s==UNROLL_X-1) prefetch_i_right(i_off, sidx, idx, smem, d_u1);
            prefetch_reg_j_down(j, idx, ys, d_u1);
            prefetch_reg_j_up(j, idx, ys, d_u1);
            ys[STENCIL_DEPTH] = d_u1[idx];
            smem[sidx] = ys[STENCIL_DEPTH];
        }
    }
    this_thread_block().sync();
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        i_off = i + ioff;
        idx = i_off + j*NX;
        sidx = (si+ioff) + threadIdx.y*SMEM_P_X;
        ys = yval[s];
        if (check_stencil_border_2d(i_off, j, jstart, jend))
        {
            d_u2[idx] = smem_reg_stencil(smem, ys, sidx) / STENCIL_COEFF - ys[STENCIL_DEPTH];
        }
    }
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
    prefetch(smem, d_u1, 0, i, j, idx, sidx, jstart, jend);
    this_thread_block().sync();
    apply_stencil_prefetched(smem, d_u2, i, j, idx, sidx, jstart, jend);
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
        if (check_domain_border_2d(i+ioff, j, jstart, jend))
            smem[threadIdx.y][threadIdx.x+ioff] = d_u1[idx];
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
            if (s>0)          accumulate_l(&u, *smem, sidx, 1);
            else              accumulate_l(&u, *smem, d_u1, sidx, idx, threadIdx.x, 1, 1);
            if (s+1<UNROLL_X) accumulate_r(&u, *smem, sidx, 1);
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
        prefetch(smem, d_u1, s, i+ioff, j, idx, sidx, jstart, jend);
    }
    this_thread_block().sync();
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        idx = (i+ioff) + j*NX;
        sidx = (si+ioff) + sj*SMEM_P_X;
        apply_stencil_prefetched(smem, d_u2, i+ioff, j, idx, sidx, jstart, jend);
    }
}

__global__ void gpu_stencil_smem_1d(float* __restrict__ d_u1,
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
