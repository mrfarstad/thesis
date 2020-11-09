#include <stdio.h>

#ifndef NX
#define NX 256
#endif

#ifndef NY
#define NY 256
#endif


void saveResult(float *d_u)
{
    char fname[15];
    sprintf(fname, "result");

    FILE *fp_snap = fopen(fname, "w");

    fwrite(d_u, sizeof(float), NX * NY, fp_snap);
    printf("Saving %s: nx = %d ny = %d\n", fname, NX, NY);
    fflush(stdout);
    fclose(fp_snap);
    return;
}
