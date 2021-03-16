#include "../include/constants.h"
#include "stencils.cu"

__global__ void gpu_stencil_base(float* __restrict__ d_u1,
			         float* __restrict__ d_u2,
                                 unsigned int kstart,
                                 unsigned int kend)
{
    unsigned int   i, j, k, idx;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    k  = threadIdx.z + blockIdx.z*BLOCK_Z;
    idx = i + j*NX + k*NX*NY;
    if (i>=STENCIL_DEPTH && i<NX-STENCIL_DEPTH &&
        j>=STENCIL_DEPTH && j<NY-STENCIL_DEPTH &&
        k>=kstart+STENCIL_DEPTH && k<=kend-STENCIL_DEPTH)
        d_u2[idx] = stencil(d_u1, idx);
}

