#ifndef STENCIL_INITIALIZER_H
#define STENCIL_INITIALIZER_H

#include "constants.h"

void initialize_host_region(float *h_u)
{
    unsigned long i, j, k, idx;
    for (k=0; k<NZ; k++) {
        for (j=0; j<NY; j++) {
            for (i=0; i<NX; i++) {
                idx = i + j*NX + k*NX*NY;
                if (i==0 || i==NX-1 || j==0 || j==NY-1 || k==0 || k==NZ-1)
                  h_u[idx] = 1.0f;           // Dirichlet b.c.'s
                else
                  h_u[idx] = 0.0f;
            }
        }
    }
}

#endif
