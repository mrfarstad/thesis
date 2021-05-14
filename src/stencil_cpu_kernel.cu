#include "../include/constants.h"
#include "stencils.cu"

void cpu_stencil_3d(float* u1, float* u2) 
{
    unsigned long i, j, k;
    for (k=STENCIL_DEPTH; k<NZ-STENCIL_DEPTH; k++)
        for (j=STENCIL_DEPTH; j<NY-STENCIL_DEPTH; j++)
            for (i=STENCIL_DEPTH; i<NX-STENCIL_DEPTH; i++)
                stencil(u1, u2, i + j*NX + k*NX*NY);
}

void cpu_stencil_2d(float* u1, float* u2) 
{
    unsigned long i, j;
    for (j=STENCIL_DEPTH; j<NY-STENCIL_DEPTH; j++)
        for (i=STENCIL_DEPTH; i<NX-STENCIL_DEPTH; i++)
            stencil(u1, u2, i + j*NX);
}

void cpu_stencil(float* u1, float* u2) 
{
    if (DIMENSIONS==3) return cpu_stencil_3d(u1, u2);
    else return cpu_stencil_2d(u1, u2);
}
