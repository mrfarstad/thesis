#include "../include/constants.h"
#include "stencils.cu"

void cpu_stencil_3d(float* u1, float* u2) 
{
    unsigned long i, j, k;
    for (k=RADIUS; k<NZ-RADIUS; k++)
        for (j=RADIUS; j<NY-RADIUS; j++)
            for (i=RADIUS; i<NX-RADIUS; i++)
                stencil(u1, u2, i + j*NX + k*NX*NY);
}

void cpu_stencil_2d(float* u1, float* u2) 
{
    unsigned long i, j;
    for (j=RADIUS; j<NY-RADIUS; j++)
        for (i=RADIUS; i<NX-RADIUS; i++)
            stencil(u1, u2, i + j*NX);
}

void cpu_stencil(float* u1, float* u2) 
{
    if (DIMENSIONS==3) return cpu_stencil_3d(u1, u2);
    else return cpu_stencil_2d(u1, u2);
}
