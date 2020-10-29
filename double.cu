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

    // set up gpu card
    int *d_u1[ngpus];

    size_t isize = 5;
    size_t ibyte = isize * sizeof(int);

    int *host_ref = (int *) calloc(isize * ngpus, sizeof(int));

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((void **) &d_u1[i], ibyte));
    }

    dim3 block(isize);
    dim3 grid(1);

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        //test<<<grid, block>>>(d_u1[i], isize);
        void *kernelArgs[] = { 
            &d_u1[i]
        };
        cudaLaunchCooperativeKernel((void*) test, grid, block, kernelArgs);
    }

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemcpy(&host_ref[isize * i], d_u1[i], ibyte, cudaMemcpyDeviceToHost));
    }

    for (int d = 0; d < ngpus; d++) {
        for (int i = 0; i < isize; i++) {
            printf("%d ", host_ref[i]);
        }
        printf("\n");
    }

    CHECK(cudaDeviceReset());
}
