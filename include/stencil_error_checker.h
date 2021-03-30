#include <stdio.h>
#include <math.h>
#include "constants.h"
#include <stdbool.h>

void check_domain_errors_3d(float *h_u1, float *h_u2, bool *error)

{
    unsigned long i, j, k, idx;
    double err = 0.0;
    for (k=0; k<NZ; k++) {
        for (j=0; j<NY; j++) {
            for (i=0; i<NX; i++) {
                idx = i + j*NX + k*NX*NY;
                err += (h_u1[idx]-h_u2[idx])*(h_u1[idx]-h_u2[idx]);
            }
        }
    }
    printf("rms error = %f \n",sqrt(err/ (double)SIZE));
    *error = err > 10e-6;
}

void check_domain_errors_2d(float *h_u1, float *h_u2, bool *error)

{
    unsigned long i, j, idx;
    double err = 0.0;
    for (j=0; j<NY; j++) {
        for (i=0; i<NX; i++) {
            idx = i + j*NX;
            err += (h_u1[idx]-h_u2[idx])*(h_u1[idx]-h_u2[idx]);
        }
    }
    printf("rms error = %f \n",sqrt(err/ (double)SIZE));
    *error = err > 10e-6;
}

void check_domain_errors_1d(float *h_u1, float *h_u2, bool *error)
{
    unsigned long i;
    double err = 0.0;
    for (i=0; i<NX; i++) {
        err += (h_u1[i]-h_u2[i])*(h_u1[i]-h_u2[i]);
    }
    printf("rms error = %f \n",sqrt(err/ (double)SIZE));
    *error = err > 10e-6;
}

void check_domain_errors(float *h_u1, float *h_u2, bool* error)
{
    if      (DIMENSIONS==3) return check_domain_errors_3d(h_u1, h_u2, error);
    else if (DIMENSIONS==2) return check_domain_errors_2d(h_u1, h_u2, error);
    else                    return check_domain_errors_1d(h_u1, h_u2, error);
}
