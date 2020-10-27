#include "common.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern void saveResult(
    int nx,
    int ny,
    int nz,
    float *results)
{
    unsigned int isize = nx * ny * nz;

    char fname[7];
    sprintf(fname, "result");

    FILE *fp_snap = fopen(fname, "w");

    fwrite(results, sizeof(float), isize, fp_snap);
    printf("%s: nx = %d ny = %d nz = %d\n", fname, nx, ny, nz);
    fflush(stdout);
    fclose(fp_snap);

    return;
}
