#ifndef STENCILS_BORDER_CHECK_CU
#define STENCILS_BORDER_CHECK_CU

#include "../include/constants.h"

__device__ bool check_stencil_border_1d(unsigned int i)
{
    return i>=STENCIL_DEPTH && i<NX-STENCIL_DEPTH;
}

__device__ bool check_stencil_border_1d(unsigned int i, unsigned int istart, unsigned int iend)
{
#if NGPUS>1
    return i>=istart+STENCIL_DEPTH && i<=iend-STENCIL_DEPTH;
#endif
    return check_stencil_border_1d(i);
}

__device__ bool check_stencil_border_2d(unsigned int i, unsigned int j)
{
    return check_stencil_border_1d(i) && j>=STENCIL_DEPTH && j<NY-STENCIL_DEPTH;
}

__device__ bool check_stencil_border_2d(unsigned int i, unsigned int j, unsigned int jstart, unsigned int jend)
{
#if NGPUS>1
    return check_stencil_border_1d(i) && j>=jstart+STENCIL_DEPTH && j<=jend-STENCIL_DEPTH;
#endif
    return check_stencil_border_2d(i, j);
}

__device__ bool check_stencil_border_3d(unsigned int i, unsigned int j, unsigned int k)
{
    return check_stencil_border_2d(i, j) && k>=STENCIL_DEPTH && k<NZ-STENCIL_DEPTH;
}

__device__ bool check_stencil_border_3d(
        unsigned int i, unsigned int j, unsigned int k, unsigned int kstart, unsigned int kend)
{
#if NGPUS>1
    return check_stencil_border_2d(i, j) && k>=kstart+STENCIL_DEPTH && k<=kend-STENCIL_DEPTH;
#endif
    return check_stencil_border_3d(i, j, k);
}


__device__ bool check_domain_border_1d(unsigned int i)
{
    return i<NX;
}

__device__ bool check_domain_border_1d(unsigned int i, unsigned int istart, unsigned int iend)
{
#if NGPUS>1
    return i>=istart && i<=iend;
#endif
    return check_domain_border_1d(i);
}

__device__ bool check_domain_border_2d(unsigned int i, unsigned int j)
{
    return check_domain_border_1d(i) && j<NY;
}

__device__ bool check_domain_border_2d(unsigned int i, unsigned int j, unsigned int jstart, unsigned int jend)
{
#if NGPUS>1
    return check_domain_border_1d(i) && j>=jstart && j<=jend;
#endif
    return check_domain_border_2d(i, j);
}

__device__ bool check_domain_border_3d(unsigned int i, unsigned int j, unsigned int k)
{
    return check_domain_border_2d(i, j) && k<NZ;
}

__device__ bool check_domain_border_3d(
        unsigned int i, unsigned int j, unsigned int k, unsigned int kstart, unsigned int kend)
{
#if NGPUS>1
    return check_domain_border_2d(i, j) && k>=kstart && k<=kend;
#endif
    return check_domain_border_3d(i, j, k);
}

#endif // STENCILS_BORDER_CHECK_CU
