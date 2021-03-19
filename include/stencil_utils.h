#ifndef STENCIL_UTILS_H
#define STENCIL_UTILS_H

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include "constants.h"

#define STRLEN 35
#define STR_INDIR(x) #x
#define STR(x) STR_INDIR(x)
#define SUFFIX "_" STR(DIM) "_" STR(ITERATIONS) "_" STR(STENCIL_DEPTH) "_" STR(DIMENSIONS) "d"
#define SOLUTION "solutions/solution" SUFFIX
#define RESULT "results/result" SUFFIX

bool file_exists (char *filename) {
  struct stat   buffer;   
  return (stat (filename, &buffer) == 0);
}

static void save(float *d_u, char const *format)
{
    char fname[STRLEN];
    sprintf(fname, "%s", format);

    FILE *fp_snap = fopen(fname, "w");
    printf("saving %s: nx = %d ny = %d nz = %d iterations: %d\n", fname, NX, (unsigned int) NY,  NZ, ITERATIONS);
    fwrite(d_u, sizeof(float), SIZE, fp_snap);
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
    struct stat st = {0};
    if (stat("results", &st) == -1) {
        mkdir("results", 0700);
    }
    save(d_u, RESULT);
}

static inline void ignore_result(long long int unused_result) {
    (void) unused_result;
}

int readSolution(float *h_u)
{
    char fname[STRLEN];
    sprintf(fname, "%s", SOLUTION);

    FILE *fp = fopen(fname, "r");
    if (fp != NULL) {
        fseek(fp, 0, SEEK_SET);
        ignore_result(fread(h_u, sizeof(float), SIZE, fp));
        printf("reading %s: nx = %d ny = %d nz = %d\n", fname, NX, NY, (unsigned int) NZ);
        fclose(fp);
    }
    else {
        printf("Unable to load file: %s\n", SOLUTION);
        exit(EXIT_FAILURE);
    }
    return 0;
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
    printf("Version: ");
    if      (COOP && SMEM) printf("coop_smem");
    else if (SMEM)         printf("smem");
    else if (COOP)         printf("coop_base");
    else                   printf("base");
    printf(" (DIMENSIONS=%d DIM=%d BLOCK_X=%d BLOCK_Y=%d BLOCK_Z=%d NGPUS=%d STENCIL_DEPTH=%d)", DIMENSIONS, DIM, BLOCK_X, BLOCK_Y, BLOCK_Z, NGPUS, STENCIL_DEPTH);
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

#endif
