#include "common.h"
#include "utils.h"
#include <stdio.h>
#include "cooperative_groups.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

using namespace cooperative_groups;

__global__
void test(int *d1, int *d2, int device)
{
    multi_grid_group mgg = this_multi_grid();
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d1[idx] = idx * device;
    mgg.sync();
    memcpy(d2, d1, 5 * sizeof(int));//, cudaMemcpyDeviceToDevice);
    // Race-condition faktisk...
}

int main(int argc, char *argv[]) {

    int ngpus = 4;

    ENABLE_P2P(ngpus);

    cudaLaunchParams *launchParams = (cudaLaunchParams*) malloc(sizeof(cudaLaunchParams) * ngpus);
    cudaStream_t     *streams      = (cudaStream_t*)     malloc(sizeof(cudaStream_t)     * ngpus);

    // set up gpu card
    int *d_u[ngpus];

    size_t isize = 5;
    size_t ibyte = isize * sizeof(int);

    int *host_ref;
    cudaMallocHost((void **) &host_ref, ibyte * ngpus);

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamCreate(&streams[i]));
        //CHECK(cudaMemcpyToSymbol(device, &i, sizeof(int)));
        //int d;
        //cudaGetDevice(&d);
        //CHECK(cudaMemcpyToSymbol(device, &i, sizeof(int)));
        //CHECK(cudaMemcpyToSymbolAsync(device, &i, sizeof(int), 0, cudaMemcpyHostToDevice, streams[i]));
    }



    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((void **) &d_u[i], ibyte));
    }

    CHECK(cudaSetDevice(0));
    dim3 block(isize);
    dim3 grid(1);


    void *args[ngpus][3];
    int devices[ngpus];

    for (int i = 0; i < 4; i++)
    {
        devices[i] = i;
    }

    for (int i = 0; i < ngpus; i++)
    {
        args[i][0] = &d_u[i];
        args[i][1] = &d_u[(i+1)%ngpus];
        args[i][2] = &devices[i];
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
        CHECK(cudaMemcpyAsync(&host_ref[i * isize], d_u[i], ibyte, cudaMemcpyDeviceToHost, streams[i]));
    }

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamDestroy(streams[i]));
        CHECK(cudaFree(d_u[i]));
    }

    for (int d = 0; d < ngpus; d++) {
        for (int h = 0; h < isize; h++) {
            printf("%d ", host_ref[h]);
        }
        printf("\n");
    }

    CHECK(cudaDeviceReset());

    cudaFree(host_ref);
    free(launchParams);
    free(streams);
}
