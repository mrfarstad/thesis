#include "../include/constants.h"
#include "stencils.cu"

__device__ inline void calculateBaseStencil(float* d_u1, float*d_u2, unsigned int i)
{
    if (i>=STENCIL_DEPTH && i<NX-STENCIL_DEPTH)
        d_u2[i] = stencil(d_u1, i);
}

__device__ inline void calculateBaseStencil(float* d_u1, float*d_u2, unsigned int i, unsigned int j)
{
    unsigned int idx = i + j*NX;
    if (i>=STENCIL_DEPTH && i<NX-STENCIL_DEPTH &&
        j>=STENCIL_DEPTH && j<NY-STENCIL_DEPTH)
        d_u2[idx] = stencil(d_u1, idx);
}

__device__ inline void calculateBaseStencil(float* d_u1, float*d_u2, unsigned int i, unsigned int j, unsigned int k)
{
    unsigned int idx = i + j*NX + k*NX*NY;
    if (i>=STENCIL_DEPTH && i<NX-STENCIL_DEPTH &&
        j>=STENCIL_DEPTH && j<NY-STENCIL_DEPTH &&
        k>=STENCIL_DEPTH && k<NZ-STENCIL_DEPTH)
        d_u2[idx] = stencil(d_u1, idx);
}

__global__ void gpu_stencil_base_1d(float* __restrict__ d_u1,
			            float* __restrict__ d_u2,
                                    unsigned int kstart,
                                    unsigned int kend)
{
    calculateBaseStencil(d_u1, d_u2, threadIdx.x + blockIdx.x*BLOCK_X);
}

__global__ void gpu_stencil_base_1d_unrolled(float* __restrict__ d_u1,
                                             float* __restrict__ d_u2,
                                             unsigned int kstart,
                                             unsigned int kend)
{
    unsigned int i, u;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
#pragma unroll
    for (u=0; u<UNROLL_X; u++)
        calculateBaseStencil(d_u1, d_u2, i+BLOCK_X*u);
}

__global__ void gpu_stencil_base_2d(float* __restrict__ d_u1,
			            float* __restrict__ d_u2,
                                    unsigned int kstart,
                                    unsigned int kend)
{
    unsigned int i, j;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    calculateBaseStencil(d_u1, d_u2, i, j);
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
        calculateBaseStencil(d_u1, d_u2, i+BLOCK_X*u, j);
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
    calculateBaseStencil(d_u1, d_u2, i, j, k);
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
        calculateBaseStencil(d_u1, d_u2, i+BLOCK_X*u, j, k);
}

// This kernel performs really bad
__global__ void gpu_stencil_base_3d_unrolled_4(float* __restrict__ d_u1,
                                               float* __restrict__ d_u2,
                                               unsigned int kstart,
                                               unsigned int kend)
{
    unsigned int i, j, k, idx, u;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    if (j>=STENCIL_DEPTH && j<NY-STENCIL_DEPTH &&
        k>=STENCIL_DEPTH && k<NZ-STENCIL_DEPTH)
    {
        idx = i + j*NX + k*NX*NY;
        if (i>=STENCIL_DEPTH)
            d_u2[idx] = stencil(d_u1, idx);
#pragma unroll
        for (u=1; u<=MAX(UNROLL_X-2, 1); u++)
            d_u2[idx + u*BLOCK_X] = stencil(d_u1, idx + u*BLOCK_X);

        if ((i+(UNROLL_X-1)*BLOCK_X)<NX-STENCIL_DEPTH) 
            d_u2[idx + (UNROLL_X-1)*BLOCK_X] = stencil(d_u1, idx + (UNROLL_X-1)*BLOCK_X);
    }
}
