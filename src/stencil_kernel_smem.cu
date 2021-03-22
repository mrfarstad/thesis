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
#if DIMENSIONS>1
#pragma unroll
    for (d=STENCIL_DEPTH; d>=1; d--) {
        if (threadIdx.y >= d)        u += smem[sidx-d*SMEM_X];
        else                         u += d_u1[idx-d*NX];
    }
#pragma unroll
    for (d=1; d<=STENCIL_DEPTH; d++) {
        if (threadIdx.y+d < BLOCK_Y) u += smem[sidx+d*SMEM_X];
        else                         u += d_u1[idx+d*NX];
    }
#endif
#if DIMENSIONS>2
#pragma unroll
    for (d=STENCIL_DEPTH; d>=1; d--) {
        if (threadIdx.z >= d)        u += smem[sidx-d*SMEM_X*BLOCK_Y];
        else                         u += d_u1[idx-d*NX*NY];
    }
#pragma unroll
    for (d=1; d<=STENCIL_DEPTH; d++) {
        if (threadIdx.z+d < BLOCK_Z) u += smem[sidx+d*SMEM_X*BLOCK_Y];
        else                         u += d_u1[idx+d*NX*NY];
    }
#endif
    return u;
}


__device__ inline void accumulate (float *src, unsigned int idx, float* u, int offset) {
#pragma unroll
    for (unsigned int d=1; d<=STENCIL_DEPTH; d++) *u += src[idx+d*offset];
}

// This kernel only uses shared memory for stencil dimensions smaller than the block dimensions
__device__ inline void smem_stencil_new(float* smem, float* d_u1, unsigned int sidx, unsigned int idx, float* u) {
    if (threadIdx.x >= STENCIL_DEPTH)        accumulate(smem, sidx, u, -1);
    else                                     accumulate(d_u1, idx, u, -1);

    if (threadIdx.x+STENCIL_DEPTH < BLOCK_X) accumulate(smem, sidx, u, 1);
    else                                     accumulate(d_u1, idx, u, 1);

#if DIMENSIONS>1
    if (threadIdx.y >= STENCIL_DEPTH)        accumulate(smem, sidx, u, -SMEM_X);
    else                                     accumulate(d_u1, idx, u, -NX);

    if (threadIdx.y+STENCIL_DEPTH < BLOCK_Y) accumulate(smem, sidx, u, SMEM_X);
    else                                     accumulate(d_u1, idx, u, NX);
#endif

#if DIMENSIONS>2
    if (threadIdx.z >= STENCIL_DEPTH)        accumulate(smem, sidx, u, -(SMEM_X*BLOCK_Y));
    else                                     accumulate(d_u1, idx, u, -(NX*NY));

    if (threadIdx.z+STENCIL_DEPTH < BLOCK_Z) accumulate(smem, sidx, u, (SMEM_X)*BLOCK_Y);
    else                                     accumulate(d_u1, idx, u, NX*NY);
#endif
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
    sidx = threadIdx.x + threadIdx.y*(BLOCK_X+SMEM_PAD) + threadIdx.z*(BLOCK_X+SMEM_PAD)*BLOCK_Y;
    if (i<NX && j<NY && k>=kstart && k<=kend) {
        u0 = d_u1[idx];
        smem[sidx] = u0;
    }
    this_thread_block().sync();
    if (i>=STENCIL_DEPTH && i<NX-STENCIL_DEPTH &&
        j>=STENCIL_DEPTH && j<NY-STENCIL_DEPTH &&
        k>=kstart+STENCIL_DEPTH && k<=kend-STENCIL_DEPTH) 
    {
        d_u2[idx] = smem_stencil(smem, d_u1, sidx, idx, u) / STENCIL_COEFF - u0;
    }
}

__global__ void gpu_stencil_smem_2d_unrolled(float* __restrict__ d_u1,
			                     float* __restrict__ d_u2,
                                             unsigned int jstart,
                                             unsigned int jend)
{
    unsigned int i, j, s, idx, ioff, d;
    float u;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    __shared__ float smem[BLOCK_Y][SMEM_X];
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        idx = (i+ioff) + j*NX;
        if ((i+ioff)<NX && j<=NY)
        {
            smem[threadIdx.y][threadIdx.x+ioff] = d_u1[idx];
        }
    }
    this_thread_block().sync();
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        idx = (i+ioff) + j*NX;
        if ((i+ioff)>=STENCIL_DEPTH && (i+ioff)<NX-STENCIL_DEPTH &&
            j>=STENCIL_DEPTH && j<NY-STENCIL_DEPTH)
        {
            u = 0.0f;
            if (s>0) {
#pragma unroll
                for (d=1; d<=STENCIL_DEPTH; d++)
                    u += smem[threadIdx.y][threadIdx.x+ioff-d];
            } else {
#pragma unroll
                for (d=1; d<=STENCIL_DEPTH; d++) {
                    if (threadIdx.x+ioff >= d)        u += smem[threadIdx.y][threadIdx.x+ioff-d];
                    else                              u += d_u1[idx-d];
                }
            }
            if (s<UNROLL_X-1){
#pragma unroll
                for (d=1; d<=STENCIL_DEPTH; d++)
                    u += smem[threadIdx.y][threadIdx.x+ioff+d];
            } else {
#pragma unroll
                for (d=1; d<=STENCIL_DEPTH; d++) {
                    if (threadIdx.x+ioff+d < BLOCK_X) u += smem[threadIdx.y][threadIdx.x+ioff+d];
                    else                              u += d_u1[idx+d];
                }
            }

#pragma unroll
            for (d=1; d<=STENCIL_DEPTH; d++) {
                if (threadIdx.y >= d)             u += smem[threadIdx.y-d][threadIdx.x+ioff];
                else                              u += d_u1[idx-d*NX];
            }

#pragma unroll
            for (d=1; d<=STENCIL_DEPTH; d++) {
                if (threadIdx.y+d < BLOCK_Y)      u += smem[threadIdx.y+d][threadIdx.x+ioff];
                else                              u += d_u1[idx+d*NX];
            }
            d_u2[idx] = u / STENCIL_COEFF - smem[threadIdx.y][threadIdx.x+ioff];
        }
    }
}


__global__ void gpu_stencil_smem_2d_unrolled_prefetch(float* __restrict__ d_u1,
                                                      float* __restrict__ d_u2,
                                                      unsigned int jstart,
                                                      unsigned int jend)
{
    unsigned int i, j, s, idx, sidx, ioff, d, si, sj;
    float u;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    //__shared__ float smem[SMEM_P_Y][SMEM_P_X];
    extern __shared__ float smem[];
    si = threadIdx.x + STENCIL_DEPTH;
    sj = threadIdx.y + STENCIL_DEPTH;
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        idx = (i+ioff) + j*NX;
        sidx = si+ioff + sj*SMEM_P_X;
        if ((i+ioff)<NX && j<=NY)
        {
            if (s==0 && threadIdx.x < STENCIL_DEPTH && i >= STENCIL_DEPTH)
            {
                smem[sidx-STENCIL_DEPTH] = d_u1[idx-STENCIL_DEPTH];
            }
            if (s==UNROLL_X-1 && threadIdx.x >= BLOCK_X-STENCIL_DEPTH && i < NX-STENCIL_DEPTH)
            {
                smem[sidx+STENCIL_DEPTH] = d_u1[idx+STENCIL_DEPTH];
            }
            if (threadIdx.y < STENCIL_DEPTH && j >= STENCIL_DEPTH)
            {
                smem[sidx-STENCIL_DEPTH*SMEM_P_X] = d_u1[idx-STENCIL_DEPTH*NX];
            }
            if (threadIdx.y >= BLOCK_Y-STENCIL_DEPTH && j < NY-STENCIL_DEPTH)
            {
                smem[sidx+STENCIL_DEPTH*SMEM_P_X] = d_u1[idx+STENCIL_DEPTH*NX];
            }
            smem[sidx] = d_u1[idx];
        }
    }
    this_thread_block().sync();
#pragma unroll
    for (s=0; s<UNROLL_X; s++) {
        ioff = s*BLOCK_X;
        idx = (i+ioff) + j*NX;
        sidx = (si+ioff) + sj*SMEM_P_X;
        if ((i+ioff)>=STENCIL_DEPTH && (i+ioff)<NX-STENCIL_DEPTH &&
            j>=STENCIL_DEPTH && j<NY-STENCIL_DEPTH)
        {
            u = 0.0f;
#pragma unroll
            for (d=1; d<=STENCIL_DEPTH; d++)
            {
                u += smem[sidx-d]
                   + smem[sidx+d]
                   + smem[sidx-d*SMEM_P_X]
                   + smem[sidx+d*SMEM_P_X];
            }
            d_u2[idx] = u / STENCIL_COEFF - smem[sidx];
        }
    }
}


__global__ void gpu_stencil_smem_2d(float* __restrict__ d_u1,
			            float* __restrict__ d_u2,
                                    unsigned int jstart,
                                    unsigned int jend)
{
    float u = 0.0f, u0;
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
    if (i>=STENCIL_DEPTH && i<NX-STENCIL_DEPTH &&
        j>=jstart+STENCIL_DEPTH && j<=jend-STENCIL_DEPTH) 
    {
        //d_u2[idx] = smem_stencil(smem, d_u1, sidx, idx, u) / STENCIL_COEFF - u0;
        smem_stencil_new(smem, d_u1, sidx, idx, &u);
        d_u2[idx] = u / STENCIL_COEFF - u0;
    }
}

__global__ void gpu_stencil_smem_1d(float* __restrict__ d_u1,
			            float* __restrict__ d_u2,
                                    unsigned int istart,
                                    unsigned int iend)
{
    float u = 0.0f, u0;
    unsigned int i;
    extern __shared__ float smem[];
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    if (i>=istart && i<=iend)
    {
        u0 = d_u1[i];
        smem[threadIdx.x] = u0;
    }
    this_thread_block().sync();
    if (i>=STENCIL_DEPTH && i<NX-STENCIL_DEPTH)
    {
        d_u2[i] = smem_stencil(smem, d_u1, threadIdx.x, i, u) / STENCIL_COEFF - u0;
    }
}
