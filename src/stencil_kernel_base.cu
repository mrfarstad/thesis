#include "../include/constants.h"
#include "stencils.cu"
#include "stencils_border_check.cu"

__device__ inline void calculateBaseStencil(float* __restrict__ d_u1,
                                            float* __restrict__ d_u2,
                                            unsigned int i,
                                            unsigned int istart,
                                            unsigned int iend)
{
    if (check_stencil_border_1d(i, istart, iend))
        d_u2[i] = stencil(d_u1, i);
}

__device__ inline void calculateBaseStencil(float* __restrict__ d_u1,
                                            float* __restrict__ d_u2,
                                            unsigned int i,
                                            unsigned int j,
                                            unsigned int jstart,
                                            unsigned int jend)
{
    unsigned int idx = i + j*NX;
    if (check_stencil_border_2d(i, j, jstart, jend))
        d_u2[idx] = stencil(d_u1, idx);
}

__device__ inline void calculateBaseStencil(float* __restrict__ d_u1,
                                            float* __restrict__ d_u2,
                                            unsigned int i,
                                            unsigned int j,
                                            unsigned int k,
                                            unsigned int kstart,
                                            unsigned int kend)
{
    unsigned int idx = i + j*NX + k*NX*NY;
    if (check_stencil_border_3d(i, j, k, kstart, kend))
        d_u2[idx] = stencil(d_u1, idx);
}

__global__ void gpu_stencil_base_3d(float* __restrict__ d_u1,
			            float* __restrict__ d_u2,
                                    unsigned int kstart,
                                    unsigned int kend)
{
    unsigned int i, j, k;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    calculateBaseStencil(d_u1, d_u2, i, j, k, kstart, kend);
}

__global__ void gpu_stencil_base_3d_unrolled(float* __restrict__ d_u1,
                                             float* __restrict__ d_u2,
                                             unsigned int kstart,
                                             unsigned int kend)
{
    unsigned int i, j, k, u;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
#pragma unroll
    for (u=0; u<UNROLL_X; u++)
        calculateBaseStencil(d_u1, d_u2, i+BLOCK_X*u, j, k, kstart, kend);
}

__global__ void gpu_stencil_base_2d(float* __restrict__ d_u1,
			            float* __restrict__ d_u2,
                                    unsigned int jstart,
                                    unsigned int jend)
{
    unsigned int i, j;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    calculateBaseStencil(d_u1, d_u2, i, j, jstart, jend);
}

__global__ void gpu_stencil_base_2d_unrolled(float* __restrict__ d_u1,
                                             float* __restrict__ d_u2,
                                             unsigned int kstart,
                                             unsigned int kend)
{
    unsigned int i, j, u;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
#pragma unroll
    for (u=0; u<UNROLL_X; u++)
        calculateBaseStencil(d_u1, d_u2, i+BLOCK_X*u, j, kstart, kend);
}

__global__ void gpu_stencil_base_1d(float* __restrict__ d_u1,
			            float* __restrict__ d_u2,
                                    unsigned int istart,
                                    unsigned int iend)
{
    calculateBaseStencil(d_u1, d_u2, threadIdx.x + blockIdx.x*BLOCK_X, istart, iend);
}

__global__ void gpu_stencil_base_1d_unrolled(float* __restrict__ d_u1,
                                             float* __restrict__ d_u2,
                                             unsigned int istart,
                                             unsigned int iend)
{
    unsigned int i, u;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
#pragma unroll
    for (u=0; u<UNROLL_X; u++)
        calculateBaseStencil(d_u1, d_u2, i+BLOCK_X*u, istart, iend);
}
