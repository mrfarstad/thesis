#ifndef STENCILS_CU
#define STENCILS_CU

#include "../include/constants.h"

__device__ bool check_stencil_border_1d(unsigned int i)
{
    return i>=STENCIL_DEPTH && i<NX-STENCIL_DEPTH;
}

__device__ bool check_stencil_border_2d(unsigned int i, unsigned int j)
{
    return check_stencil_border_1d(i) && j>=STENCIL_DEPTH && j<NY-STENCIL_DEPTH;
}

__device__ bool check_stencil_border_3d(unsigned int i, unsigned int j, unsigned int k)
{
    return check_stencil_border_2d(i, j) && k>=STENCIL_DEPTH && k<NZ-STENCIL_DEPTH;
}

__host__ __device__ float stencil(float *d_u1, unsigned int idx) {
    float u = 0.0f, c = (float) (2*DIMENSIONS*STENCIL_DEPTH);
    unsigned int dim, s;
    unsigned int offsets[DIMENSIONS] = {1};
#if DIMENSIONS>1
    offsets[1] = NX;
#endif
#if DIMENSIONS>2
    offsets[2] = NX*NY;
#endif
    for (dim=0; dim<DIMENSIONS; dim++)
        for (s=1; s<=STENCIL_DEPTH; s++)
            u+=(d_u1[idx-s*offsets[dim]] + d_u1[idx+s*offsets[dim]]);
    return u / c - d_u1[idx];
}


__device__ __inline__ void accumulate_l(float *u, float *smem, float *d_u1, unsigned int sidx,
                                        unsigned int idx, int soffset, int offset)
{
#pragma unroll
    for (unsigned int d=1; d<=STENCIL_DEPTH; d++) *u += smem[sidx-d*soffset];
}

__device__ __inline__ void accumulate_l_l(float *u, float *smem, float *d_u1, unsigned int sidx,
                                          unsigned int idx, unsigned int t,
                                          int soffset, int offset)
{
#pragma unroll
    for (unsigned int d=1; d<=STENCIL_DEPTH; d++) *u += (t >= d) ? smem[sidx-d*soffset] : d_u1[idx-d*offset];
}

__device__ __inline__ void accumulate_l(float *u, float *smem, float *d_u1, unsigned int sidx,
                                        unsigned int idx, unsigned int t,
                                        int soffset, int offset)
{
#pragma unroll
    for (unsigned int d=STENCIL_DEPTH; d>=1; d--) *u += (t >= d) ? smem[sidx-d*soffset] : d_u1[idx-d*offset];
}

__device__ __inline__ void accumulate_r(float *u, float *smem, float *d_u1, unsigned int sidx,
                                        unsigned int idx, int soffset, int offset)
{
#pragma unroll
    for (unsigned int d=1; d<=STENCIL_DEPTH; d++) *u += smem[sidx+d*soffset];
}

__device__ __inline__ void accumulate_r(float *u, float *smem, float *d_u1, unsigned int sidx,
                             unsigned int idx, unsigned int t, unsigned int l,
                             int soffset, int offset)
{
#pragma unroll
    for (unsigned int d=1; d<=STENCIL_DEPTH; d++) *u += (t+d < l) ? smem[sidx+d*soffset] : d_u1[idx+d*offset];
}

__device__ float smem_stencil(float* smem, float* d_u1, unsigned int sidx, unsigned int idx)
{
    float u = 0.0f;
    accumulate_l(&u, smem, d_u1, sidx, idx, threadIdx.x, 1, 1);
    accumulate_r(&u, smem, d_u1, sidx, idx, threadIdx.x, BLOCK_X, 1, 1);
#if DIMENSIONS>1
    accumulate_l(&u, smem, d_u1, sidx, idx, threadIdx.y, SMEM_X, NX);
    accumulate_r(&u, smem, d_u1, sidx, idx, threadIdx.y, BLOCK_Y, SMEM_X, NX);
#endif
#if DIMENSIONS>2
    accumulate_l(&u, smem, d_u1, sidx, idx, threadIdx.z, SMEM_X*BLOCK_Y, NX*NY);
    accumulate_r(&u, smem, d_u1, sidx, idx, threadIdx.z, BLOCK_Z, SMEM_X*BLOCK_Y, NX*NY);
#endif
    return u;
}


__device__ void apply_stencil_prefetched(
        unsigned int i,
        unsigned int j,
        unsigned int idx,
        unsigned int sidx,
        float *smem,
        float *d_u2)
{
    float u = 0.0f;
    unsigned int d;
    if (check_stencil_border_2d(i, j))
    {
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

#endif // STENCILS_CU
