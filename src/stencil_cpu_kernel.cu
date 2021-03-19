#include "../include/constants.h"
#include "stencils.cu"

void cpu_stencil_3d(float* u1, float* u2) 
{
  unsigned long i, j, k, ind;
  for (k=STENCIL_DEPTH; k<NZ-STENCIL_DEPTH; k++) {
    for (j=STENCIL_DEPTH; j<NY-STENCIL_DEPTH; j++) {
      for (i=STENCIL_DEPTH; i<NX-STENCIL_DEPTH; i++) {
        ind = i + j*NX + k*NX*NY;
        u2[ind] = stencil(u1, ind);
      }
    }
  }
}

void cpu_stencil_2d(float* u1, float* u2) 
{
  unsigned long i, j, ind;
  for (j=STENCIL_DEPTH; j<NY-STENCIL_DEPTH; j++) {
    for (i=STENCIL_DEPTH; i<NX-STENCIL_DEPTH; i++) {
      ind = i + j*NX;
      u2[ind] = stencil(u1, ind);
    }
  }
}

void cpu_stencil_1d(float* u1, float* u2) 
{
  unsigned long i;
  for (i=STENCIL_DEPTH; i<NX-STENCIL_DEPTH; i++)
    u2[i] = stencil(u1, i);
}

void cpu_stencil(float* u1, float* u2) 
{
    if      (DIMENSIONS==3) return cpu_stencil_3d(u1, u2);
    else if (DIMENSIONS==2) return cpu_stencil_2d(u1, u2);
    else                    return cpu_stencil_1d(u1, u2);
}
