#include "../include/constants.h"
#include "stencils.cu"

__device__ inline void calculateStencil(float* d_u1, float*d_u2, unsigned int i, unsigned int j, unsigned int k)
{
    unsigned int idx = i + j*NX + k*NX*NY;
    if (i>=STENCIL_DEPTH && i<NX-STENCIL_DEPTH &&
        j>=STENCIL_DEPTH && j<NY-STENCIL_DEPTH &&
        k>=STENCIL_DEPTH && k<NZ-STENCIL_DEPTH)
        d_u2[idx] = stencil(d_u1, idx);
}

__global__ void gpu_stencil_base(float* __restrict__ d_u1,
			         float* __restrict__ d_u2,
                                 unsigned int kstart,
                                 unsigned int kend)
{
    unsigned int   i, j, k;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    calculateStencil(d_u1, d_u2, i, j, k);
}

__global__ void gpu_stencil_base_unrolled(float* __restrict__ d_u1,
                                          float* __restrict__ d_u2,
                                          unsigned int kstart,
                                          unsigned int kend)
{
    unsigned int   i, j, k, u;
    i  = threadIdx.x + blockIdx.x*BLOCK_X*UNROLL_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
#pragma unroll
    for (u=0; u<UNROLL_X; u++)
        calculateStencil(d_u1, d_u2, i+BLOCK_X*u, j, k);
}

__global__ void gpu_stencil_base_unrolled_4(float* __restrict__ d_u1,
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
