#ifndef STENCIL_INITIALIZER_H
#define STENCIL_INITIALIZER_H

#include "constants.h"

void initialize_host_region_3d(float *h_u)
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

void initialize_host_region_2d(float *h_u)
{
    unsigned long i, j, idx;
    for (j=0; j<NY; j++) {
        for (i=0; i<NX; i++) {
            idx = i + j*NX;
            if (i==0 || i==NX-1 || j==0 || j==NY-1)
              h_u[idx] = 1.0f;           // Dirichlet b.c.'s
            else
              h_u[idx] = 0.0f;
        }
    }
}

void initialize_host_region_1d(float *h_u)
{
    unsigned long i;
    for (i=0; i<NX; i++) {
        if (i==0 || i==NX-1)
          h_u[i] = 1.0f;           // Dirichlet b.c.'s
        else
          h_u[i] = 0.0f;
    }
}

void initialize_host_region(float *h_u)
{
    if      (DIMENSIONS==3) return initialize_host_region_3d(h_u);
    else if (DIMENSIONS==2) return initialize_host_region_2d(h_u);
    else                    return initialize_host_region_1d(h_u);
}

#endif
