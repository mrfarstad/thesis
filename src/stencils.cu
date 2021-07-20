#ifndef STENCILS_CU
#define STENCILS_CU

#include "../include/constants.h"
#include "stencil_border_check.cu"
#include "stencil_accumulators.cu"

__device__ __host__  __inline__ void stencil(float *d_u1, float *d_u2, unsigned int idx) {
    float u = 0.0f;
    accumulate_global_i_prev(&u, d_u1, idx);
    accumulate_global_i_next(&u, d_u1, idx);
    accumulate_global_j_prev(&u, d_u1, idx);
    accumulate_global_j_next(&u, d_u1, idx);
#if DIMENSIONS>2
    accumulate_global_k_prev(&u, d_u1, idx);
    accumulate_global_k_next(&u, d_u1, idx);
#endif
    d_u2[idx] = u / STENCIL_COEFF - d_u1[idx];
}

__device__ __inline__ void smem_stencil(float* smem, float* d_u1, float* d_u2, unsigned int sidx, unsigned int idx)
{
    float u = 0.0f;
    accumulate_hybrid_i_prev(&u, smem, d_u1, sidx, idx);
    accumulate_hybrid_i_next(&u, smem, d_u1, sidx, idx);
    accumulate_hybrid_j_prev(&u, smem, d_u1, sidx, idx);
    accumulate_hybrid_j_next(&u, smem, d_u1, sidx, idx);
#if DIMENSIONS>2
    accumulate_hybrid_k_prev(&u, smem, d_u1, sidx, idx);
    accumulate_hybrid_k_next(&u, smem, d_u1, sidx, idx);
#endif
    d_u2[idx] = u / STENCIL_COEFF - smem[sidx];
}

__device__ __inline__ void smem_unrolled_stencil(
        float *d_u1,
        float *d_u2,
        float *smem,
        unsigned int s,
        unsigned int idx,
        unsigned int sidx)
{
    float u = 0.0f;
    if (s>0)          accumulate_smem_i_prev(&u, smem, sidx);
    else              accumulate_hybrid_i_prev(&u, smem, d_u1, sidx, idx);
    if (s+1<COARSEN_X) accumulate_smem_i_next(&u, smem, sidx);
    else              accumulate_hybrid_i_next(&u, smem, d_u1, sidx, idx);
    accumulate_hybrid_j_prev(&u, smem, d_u1, sidx, idx);
    accumulate_hybrid_j_next(&u, smem, d_u1, sidx, idx);
#if DIMENSIONS>2
    accumulate_hybrid_k_prev(&u, smem, d_u1, sidx, idx);
    accumulate_hybrid_k_next(&u, smem, d_u1, sidx, idx);
#endif
    d_u2[idx] = u / STENCIL_COEFF - smem[sidx];
}

__device__ __inline__ void smem_padded_stencil(float *smem, float *d_u2, unsigned int idx, unsigned int sidx)
{
    float u = 0.0f;
    accumulate_smem_i_prev(&u, smem, sidx);
    accumulate_smem_i_next(&u, smem, sidx);
    accumulate_smem_j_prev(&u, smem, sidx);
    accumulate_smem_j_next(&u, smem, sidx);
#if DIMENSIONS>2
    accumulate_smem_k_prev(&u, smem, sidx);
    accumulate_smem_k_next(&u, smem, sidx);
#endif
    d_u2[idx] = u / STENCIL_COEFF - smem[sidx];
}

__device__ __inline__ void smem_register_stencil(float* smem, float* d_u2, float* yval, unsigned int sidx, unsigned int idx)
{
    float u = 0.0f;
    accumulate_smem_i_prev(&u, smem, sidx);
    accumulate_smem_i_next(&u, smem, sidx);
#if DIMENSIONS>2
    accumulate_smem_j_prev(&u, smem, sidx);
    accumulate_smem_j_next(&u, smem, sidx);
#endif
    accumulate_register_prev(&u, yval);
    accumulate_register_next(&u, yval);
    d_u2[idx] = u / STENCIL_COEFF - yval[RADIUS];
}

#endif // STENCILS_CU
