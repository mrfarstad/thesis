#include "../include/constants.h"
#include "stencils.cu"

void cpu_stencil(float* u1, float* u2) 
{
  unsigned int   i, j, k, ind;
  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) { 
	ind = i + j*NX + k*NX*NY;
        if (i==0 || i==NX-1 || j==0 || j==NY-1 || k==0 || k==NZ-1)
          u2[ind] = u1[ind];
        else
          u2[ind] = stencil(u1, ind);
      }
    }
  }
}
