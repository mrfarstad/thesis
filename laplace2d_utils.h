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
    printf("saving %s: nx = %d ny = %d\n", fname, NX, NY);
    fwrite(d_u, sizeof(float), NX * NY, fp_snap);
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


void print_corners(float *h_u1, float *h_u2) {
    int i, j, ind;
    printf("DEVICE\n");
    for (j=0; j<8; j++) {
      for (i=0; i<8; i++) {
        ind = i + j*NX;
        printf(" %5.2f ", h_u2[ind]);
      }
      printf("\n");
    }

    printf("\n");

    printf("HOST\n");
    for (j=0; j<8; j++) {
      for (i=0; i<8; i++) {
        ind = i + j*NX;
        printf(" %5.2f ", h_u1[ind]);
      }
      printf("\n");
    }
}


void print_program_info() {
    printf("\nVersion: ");
    if      (COOP && SMEM) printf("coop_smem");
    else if (SMEM)         printf("smem");
    else if (COOP)         printf("coop_base");
    else                   printf("base");
    printf("\n");
}
