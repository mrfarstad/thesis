#include "../include/constants.h"
#include "stencils.cu"

void cpu_stencil(float* u1, float* u2) 
{
  unsigned int   i, j, k, ind;
  for (k=STENCIL_WIDTH; k<NZ-STENCIL_WIDTH; k++) {
    for (j=STENCIL_WIDTH; j<NY-STENCIL_WIDTH; j++) {
      for (i=STENCIL_WIDTH; i<NX-STENCIL_WIDTH; i++) {
        ind = i + j*NX + k*NX*NY;
        u2[ind] = stencil(u1, ind);
      }
    }
  }
}
