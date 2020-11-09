#include <stdio.h>

void saveResult(float *d_u, const int nx, const int ny)
{
    char fname[15];
    sprintf(fname, "result");

    FILE *fp_snap = fopen(fname, "w");

    fwrite(d_u, sizeof(float), nx * ny, fp_snap);
    printf("saving %s: nx = %d ny = %d\n", fname, nx, ny);
    fflush(stdout);
    fclose(fp_snap);
    return;
}
