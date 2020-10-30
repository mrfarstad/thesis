#include "common.h"
#include "utils.h"
#include <cstring>
#include <stdio.h>
#include "cooperative_groups.h"

using namespace cooperative_groups;

__global__
void test(
    int * __restrict__ src,
    int * __restrict__ dest,
    int * __restrict__ recv,
    const int isize,
    const int ibyte)
{
    multi_grid_group mg = this_multi_grid();
    memcpy(dest, src, ibyte);
    mg.sync();
    printf("Received ");
    for (int i = 0; i < isize; i++)
    {
        printf("%d ", recv[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {

    int ngpus = 4;

    cudaLaunchParams *launchParams = (cudaLaunchParams*) malloc(sizeof(cudaLaunchParams) * ngpus);
    cudaStream_t     *streams      = (cudaStream_t*)     malloc(sizeof(cudaStream_t)     * ngpus);
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamCreate(&streams[i]));
    }
    ENABLE_P2P(ngpus);
    CHECK(cudaSetDevice(0));

    // set up gpu card
    int *d_u[ngpus];
    int *r_u[ngpus];

    const int isize = 5;
    const int ibyte = isize * sizeof(int);

    int *host_ref = (int *) calloc(isize * ngpus, sizeof(int));
    int *gpu_ref = (int *) calloc(isize * ngpus, sizeof(int));

    for (int i = 0; i < ngpus; i++)
    {
        for (int j = 0; j < isize; j++)
        {
            int idx = i * isize + j;
            host_ref[idx] = idx;
        }
    }

    printf("HOST:\n");
    for (int d = 0; d < ngpus; d++) {
        for (int i = 0; i < isize; i++) {
            printf("%d ", host_ref[d * isize + i]);
        }
        printf("\n");
    }

    dim3 block(1);
    dim3 grid(1);

    // TODO: parallelize
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((void **) &d_u[i], ibyte));
        CHECK(cudaMemcpyAsync(d_u[i], &host_ref[i * isize], ibyte, cudaMemcpyHostToDevice, streams[i]));
        CHECK(cudaMalloc((void **) &r_u[i], ibyte));
    }
    CHECK(cudaSetDevice(0));

    void *args[ngpus][5];

    for (int i = 0; i < ngpus; i++)
    {
        args[i][0] = &d_u[i];
        args[i][1] = &r_u[(i+1)%ngpus];
        args[i][2] = &r_u[i];
        args[i][3] = (void *)&isize;
        args[i][4] = (void *)&ibyte;
        launchParams[i].func = (void*)test;
        launchParams[i].gridDim = grid;
        launchParams[i].blockDim = block;
        launchParams[i].sharedMem = 0;
        launchParams[i].stream = streams[i];
        launchParams[i].args = args[i];
    }

    cudaLaunchCooperativeKernelMultiDevice(launchParams, ngpus);

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemcpyAsync(&gpu_ref[isize * i], d_u[i], ibyte, cudaMemcpyDeviceToHost, streams[i]));
    }

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamDestroy(streams[i]));
        CHECK(cudaFree(d_u[i]));
        CHECK(cudaFree(r_u[i]));
    }

    printf("DEVICE:\n");
    for (int d = 0; d < ngpus; d++) {
        for (int i = 0; i < isize; i++) {
            printf("%d ", gpu_ref[d * isize + i]);
        }
        printf("\n");
    }

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaDeviceReset());
    }

    free(launchParams);
    free(streams);

    free(host_ref);
    free(gpu_ref);
}
