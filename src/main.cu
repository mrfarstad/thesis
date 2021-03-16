#include <stdio.h>
#include "../include/constants.h"
#include "../include/stencil_initializer.h"
#include "../include/stencil_error_checker.h"
#include "../include/stencil_utils.h"
#include "stencil_cpu.cu"
#include "stencil_dispatch.cu"
#include "omp.h"

int main(int argc, const char **argv) {
    float  *h_ref, *d_ref,
           *d_u1, *d_u2,
           milli;

    if (DEBUG) {
        h_ref = (float *)malloc(BYTES);
        char f[] = SOLUTION;
        if (!file_exists(f)) stencil_cpu();
        readSolution(h_ref);
    }

    d_ref = (float *)malloc(BYTES);

    initialize_host_region(d_ref);

    CU(cudaMalloc((void **)&d_u1, BYTES));
    CU(cudaMalloc((void **)&d_u2, BYTES));

    cudaEvent_t start, stop;
    CU(cudaEventCreate(&start));
    CU(cudaEventCreate(&stop));
    CU(cudaEventRecord(start));

    CU(cudaMemcpy(d_u1, d_ref, BYTES, cudaMemcpyHostToDevice));

    dispatch_kernels(d_u1, d_u2);

    CU(cudaMemcpy(d_ref, d_u1, BYTES, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (DEBUG) {
        check_domain_errors(h_ref, d_ref);
        free(h_ref);
    }

    print_program_info();
    printf("%.4f\n", milli); // Print execution time in ms

    free(d_ref);

    CU(cudaFree(d_u1));
    CU(cudaFree(d_u2));
    cudaDeviceReset();
}
