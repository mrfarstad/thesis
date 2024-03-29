#include "../include/constants.h"
#include "stencils.cu"
#include "stencil_border_check.cu"

__global__ void base_3d(float* __restrict__ d_u1,
                        float* __restrict__ d_u2,
                        unsigned int kstart,
                        unsigned int kend)
{
    unsigned int i, j, k, idx;
    i  = threadIdx.x + blockIdx.x*blockDim.x;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
    k  = threadIdx.z + blockIdx.z*blockDim.z;
    idx = i + j*NX + k*NX*NY;
    if (check_stencil_border_3d(i, j, k, kstart, kend))
        stencil(d_u1, d_u2, idx);
}

__global__ void base_unroll_3d(float* __restrict__ d_u1,
                               float* __restrict__ d_u2,
                               unsigned int kstart,
                               unsigned int kend)
{
    unsigned int i, j, k, i_off, idx, u;
    i  = threadIdx.x + blockIdx.x*blockDim.x*COARSEN_X;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
    k  = threadIdx.z + blockIdx.z*blockDim.z;
#pragma unroll
    for (u=0; u<COARSEN_X; u++) {
        i_off = i+u*blockDim.x;
        idx = i_off + j*NX + k*NX*NY;
        if (check_stencil_border_3d(i_off, j, k, kstart, kend))
            stencil(d_u1, d_u2, idx);
    }
}

__global__ void base_2d(float* __restrict__ d_u1,
                        float* __restrict__ d_u2,
                        unsigned int jstart,
                        unsigned int jend)
{
    unsigned int i, j, idx;
    i  = threadIdx.x + blockIdx.x*blockDim.x;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
    idx = i + j*NX;
    if (check_stencil_border_2d(i, j, jstart, jend))
        stencil(d_u1, d_u2, idx);
}

__global__ void base_unroll_2d(float* __restrict__ d_u1,
                               float* __restrict__ d_u2,
                               unsigned int kstart,
                               unsigned int kend)
{
    unsigned int i, j, i_off, idx, u;
    i  = threadIdx.x + blockIdx.x*blockDim.x*COARSEN_X;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
#pragma unroll
    for (u=0; u<COARSEN_X; u++) {
        i_off = i+u*blockDim.x;
        idx = i_off + j*NX;
        if (check_stencil_border_2d(i_off, j, kstart, kend))
            stencil(d_u1, d_u2, idx);
    }
}
