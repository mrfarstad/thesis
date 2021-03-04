#include "../include/constants.h"
#include "stencils.cu"

void cpu_stencil(float* u1, float* u2) 
{
  unsigned int   i, j, k, ind;
  for (k=STENCIL_DEPTH; k<NZ-STENCIL_DEPTH; k++) {
    for (j=STENCIL_DEPTH; j<NY-STENCIL_DEPTH; j++) {
      for (i=STENCIL_DEPTH; i<NX-STENCIL_DEPTH; i++) {
        ind = i + j*NX + k*NX*NY;
        u2[ind] = stencil(u1, ind);
      }
    }
  }
}
