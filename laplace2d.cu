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

#define CU checkCudaErrors 
#define start_timer cudaEventRecord

int main(int argc, const char **argv){
    int    ibyte = NX*NY * sizeof(float);
    float  *h_u1, *h_u2,
           *d_u1, *d_u2,
           milli;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_u1 = (float *)malloc(ibyte);
    h_u2 = (float *)malloc(ibyte);
    CU(cudaMalloc((void **)&d_u1, ibyte));
    CU(cudaMalloc((void **)&d_u2, ibyte));


    print_program_info();

    initialize_host_region(h_u1);

    start_timer(start);
    CU(cudaMemcpy(d_u1, h_u1, ibyte, cudaMemcpyHostToDevice));
    stop_timer(&start, &stop, &milli, "\ncudaMemcpyHostToDevice: %.1f (ms) \n");

    readSolution(h_u1);

    start_timer(start);
    if (COOP) dispatch_cooperative_groups_kernels(d_u1, d_u2);
    else      dispatch_kernels(d_u1, d_u2);
    stop_timer(&start, &stop, &milli, "\nKernel execution time: %.1f (ms) \n");
    
    start_timer(start);
    CU(cudaMemcpy(h_u2, d_u1, ibyte, cudaMemcpyDeviceToHost));
    stop_timer(&start, &stop, &milli, "\ncudaMemcpyDeviceToHost: %.1f (ms) \n");

    check_domain_errors(h_u1, h_u2, NX, NY);

    if (DEBUG) print_corners(h_u1, h_u2);
    if (TEST) saveResult(h_u2);

    CU(cudaFree(d_u1));
    CU(cudaFree(d_u2));
    free(h_u1);
    free(h_u2);

    cudaDeviceReset();
}
