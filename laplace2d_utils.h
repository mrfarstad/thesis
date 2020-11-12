#include <stdio.h>
#include "constants.h"

#define STRLEN 10
#define SOLUTION "solution"
#define RESULT "result"

static void save(float *d_u, char const *format)
{
    char fname[STRLEN];
    sprintf(fname, "%s", format);

    FILE *fp_snap = fopen(fname, "w");

    fwrite(d_u, sizeof(float), NX * NY, fp_snap);
    printf("saving %s: nx = %d ny = %d\n", fname, NX, NY);
    fflush(stdout);
    fclose(fp_snap);
    return;
}

void saveSolution(float *h_u) {
    save(h_u, SOLUTION);
}

void saveResult(float *d_u)
{
    save(d_u, RESULT);
}

static void read(float *h_u, char const *format)
{
    char fname[STRLEN];
    sprintf(fname, "%s", format);

    FILE *fp = fopen(fname, "r");
    if (fp != NULL) {
        fseek(fp, 0, SEEK_SET);
        fread(h_u, sizeof(float), NX * NY, fp);
        printf("reading %s: nx = %d ny = %d\n", fname, NX, NY);
        fclose(fp);
    } else {
        printf("Unable to load file: %s\n", format);
    }
    return;
}

void readSolution(float *h_u)
{
    read(h_u, SOLUTION);
}
