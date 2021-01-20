#include <stdio.h>
#include <math.h>

void check_domain_errors(float *h_u1, float *h_u2, const int nx, const int ny)
{
    int   i, j, idx;
    float err = 0.0;
    for (j=0; j<ny; j++) {
        for (i=0; i<nx; i++) {
            idx = i + j*nx;
            err += (h_u1[idx]-h_u2[idx])*(h_u1[idx]-h_u2[idx]);
        }
    }
    printf("rms error = %f \n",sqrt(err/ (float)(nx*ny)));
}
