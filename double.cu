#include "common.h"
#include "utils.h"
#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cooperative_groups.h"

__global__
void test(int *g_u1)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    g_u1[idx] = idx;
}

int main(int argc, char *argv[]) {

    int ngpus = 2;

    cudaLaunchParams *launchParams = (cudaLaunchParams*) malloc(sizeof(cudaLaunchParams) * ngpus);
    cudaStream_t     *streams      = (cudaStream_t*)     malloc(sizeof(cudaStream_t)     * ngpus);

    // set up gpu card
    int *d_u1[ngpus];

    size_t isize = 5;
    size_t ibyte = isize * sizeof(int);

    int *host_ref = (int *) calloc(isize * ngpus, sizeof(int));

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamCreate(&streams[i]));
        CHECK(cudaMalloc((void **) &d_u1[i], ibyte));
    }

    dim3 block(isize);
    dim3 grid(1);


    void *args[ngpus][1];

    for (int i = 0; i < ngpus; i++)
    {
        args[i][0] = &d_u1[i];
    }

    for (int i = 0; i < ngpus; i++)
    {
        launchParams[i].func = (void*)test;
        launchParams[i].gridDim = grid;
        launchParams[i].blockDim = block;
        launchParams[i].sharedMem = 0;
        launchParams[i].stream = streams[i];
        launchParams[i].args = args[i];
    }

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        //test<<<grid, block>>>(d_u1[i], isize);
        //cudaLaunchCooperativeKernel(
        //    launchParams[i].func,
        //    launchParams[i].gridDim,
        //    launchParams[i].blockDim,
        //    launchParams[i].args,
        //    launchParams[i].sharedMem,
        //    launchParams[i].stream
        //);
        //

        cudaLaunchCooperativeKernel(
            launchParams[i].func,
            launchParams[i].gridDim,
            launchParams[i].blockDim,
            launchParams[i].args,
            launchParams[i].sharedMem,
            launchParams[i].stream
        );
    }

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemcpyAsync(&host_ref[isize * i], d_u1[i], ibyte, cudaMemcpyDeviceToHost, streams[i]));
    }

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamDestroy(streams[i]));
    }

    for (int d = 0; d < ngpus; d++) {
        for (int i = 0; i < isize; i++) {
            printf("%d ", host_ref[i]);
        }
        printf("\n");
    }

    CHECK(cudaDeviceReset());

    free(launchParams);
    free(streams);
    free(host_ref);
}
