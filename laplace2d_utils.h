#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include "constants.h"

#define STRLEN 30
#define STR_INDIR(x) #x
#define STR(x) STR_INDIR(x)
#define SUFFIX "_" STR(DIM) "_" STR(ITERATIONS)
#define SOLUTION "solutions/solution" SUFFIX
#define RESULT "result" SUFFIX

static void save(float *d_u, char const *format)
{
    char fname[STRLEN];
    sprintf(fname, "%s", format);

    FILE *fp_snap = fopen(fname, "w");
    printf("saving %s: nx = %d ny = %d iterations: %d\n", fname, NX, NY, ITERATIONS);
    fwrite(d_u, sizeof(float), NX * NY, fp_snap);
    fflush(stdout);
    fclose(fp_snap);
    return;
}

void saveSolution(float *h_u) {
    struct stat st = {0};
    if (stat("solutions", &st) == -1) {
        mkdir("solutions", 0700);
    }
    save(h_u, SOLUTION);
}

void saveResult(float *d_u)
{
    save(d_u, RESULT);
}

void readSolution(float *h_u)
{
    char fname[STRLEN];
    sprintf(fname, "%s", SOLUTION);

    FILE *fp = fopen(fname, "r");
    if (fp != NULL) {
        fseek(fp, 0, SEEK_SET);
        fread(h_u, sizeof(float), NX * NY, fp);
        printf("reading %s: nx = %d ny = %d\n", fname, NX, NY);
        fclose(fp);
    } else {
        printf("Unable to load file: %s\n", SOLUTION);
    }
    return;
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
    printf(" (DIM=%d BLOCK_X=%d BLOCK_Y=%d NGPUS=%d)", DIM, BLOCK_X, BLOCK_Y, NGPUS);
    printf("\n");
}


/*
 * enable P2P memcopies between GPUs (all GPUs must be compute capability 2.0 or
 * later (Fermi or later))
 */                                                                             
#define ENABLE_P2P(ngpus)                                                    \
{                                                                            \
    for (int i = 0; i < ngpus; i++)                                          \
    {                                                                        \
        CU(cudaSetDevice(i));                                                \
                                                                             \
        for (int j = 0; j < ngpus; j++)                                      \
        {                                                                    \
            if (i == j) continue;                                            \
                                                                             \
            int peer_access_available = 0;                                   \
            CU(cudaDeviceCanAccessPeer(&peer_access_available, i, j));       \
                                                                             \
            if (peer_access_available) CU(cudaDeviceEnablePeerAccess(j, 0)); \
            else  printf("P2P is not available for (%d, %d)\n", i, j);       \
        }                                                                    \
    }                                                                        \
}                                                                            \

