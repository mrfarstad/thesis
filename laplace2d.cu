#include <stdio.h>
#include "constants.h"
#include "helper_cuda.h"
#include "laplace2d_initializer.h"
#include "laplace2d_error_checker.h"
#include "laplace2d_utils.h"
#include "laplace2d_timer.cu"
#include "laplace2d_dispatch.cu"
#include "cooperative_groups.h"
using namespace cooperative_groups;

int main(int argc, const char **argv){
    int offset;
    float  *h_u1, *h_u2,
           *d_u1, *d_u2,
           milli;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_u1 = (float *)malloc(BYTES);
    h_u2 = (float *)malloc(BYTES);
    CU(cudaMalloc((void **)&d_u1, BYTES));
    CU(cudaMalloc((void **)&d_u2, BYTES));

    cudaStream_t streams[STREAMS];

    for (int i = 0; i < STREAMS; i++)
    {
        CU(cudaStreamCreate( &streams[i] ));
    }

    print_program_info();

    initialize_host_region(h_u1);

    for (int i = 0; i < STREAMS; i++)
    {
        offset = i * OFFSET;
        CU(cudaMemcpyAsync(&d_u1[offset], &h_u1[offset], BYTES_PER_STREAM, cudaMemcpyHostToDevice, streams[i]));
    }

    readSolution(h_u1);

    start_timer(start);
    //if (COOP) dispatch_cooperative_groups_kernels(d_u1, d_u2);
    //else
    dispatch_kernels(d_u1, d_u2, streams);
    stop_timer(&start, &stop, &milli, "\nKernel execution time: %.1f (ms) \n");
    
    for (int i = 0; i < STREAMS; i++)
    {
        offset = i * OFFSET;
        CU(cudaMemcpyAsync(&h_u2[offset], &d_u1[offset], BYTES_PER_STREAM, cudaMemcpyDeviceToHost, streams[i]));
    }
    
    for (int i = 0; i < STREAMS; i++)
    {
        cudaDeviceSynchronize();
    }

    check_domain_errors(h_u1, h_u2, NX, NY);

    if (DEBUG) print_corners(h_u1, h_u2);
    if (TEST || DEBUG) saveResult(h_u2);

    for (int i = 0; i < STREAMS; i++)
    {
        CU(cudaStreamDestroy(streams[i]));
    }

    CU(cudaFree(d_u1));
    CU(cudaFree(d_u2));
    free(h_u1);
    free(h_u2);

    cudaDeviceReset();
}
