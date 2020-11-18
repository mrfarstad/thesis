#include <stdio.h>
#include "constants.h"
#include "helper_cuda.h"
#include "laplace2d_initializer.h"
#include "laplace2d_error_checker.h"
#include "laplace2d_utils.h"
#include "laplace2d_timer.cu"
#include "laplace2d_dispatch.cu"
#include "cooperative_groups.h"
#include "omp.h"
using namespace cooperative_groups;

int main(int argc, const char **argv) {
    float  *h_ref, *d_ref,
           *d_u1[NGPUS], *d_u2[NGPUS],
           milli;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_ref = (float *)malloc(BYTES);
    if (cudaMallocHost((void**)&d_ref, BYTES) != cudaSuccess) {
        fprintf(stderr, "Error returned from pinned host memory allocation\n");
        exit(1);
    }

    if (NGPUS>1) ENABLE_P2P(NGPUS);

    if (DEBUG) {
        print_program_info();
        initialize_host_region(d_ref);
    }

    int size = BYTES_PER_GPU;
    if (NGPUS>1) size+=BYTES_HALO;
#pragma omp parallel for
    for (int i = 0; i < NGPUS; i++) {
        cudaSetDevice(i);
        CU(cudaMalloc((void **)&d_u1[i], size));
        CU(cudaMalloc((void **)&d_u2[i], size));
    }

    cudaSetDevice(0);
    start_timer(start);

    int offset;
    if (NGPUS==1) offset=0;
    else          offset=NX;
#pragma omp parallel for
    for (int i = 0; i < NGPUS; i++) {
        cudaSetDevice(i);
        CU(cudaMemcpy(&d_u1[i][offset], &d_ref[i * OFFSET], BYTES_PER_GPU, cudaMemcpyHostToDevice));
    }

    if (DEBUG) {
        readSolution(h_ref);
    }

    if(NGPUS==1) {
        if (COOP) dispatch_cooperative_groups_kernels(d_u1[0], d_u2[0]);
        else      dispatch_kernels(d_u1[0], d_u2[0]);
    } else dispatch_multi_gpu_kernels(d_u1, d_u2);
    
#pragma omp parallel for
    for (int i = 0; i < NGPUS; i++) {
        cudaSetDevice(i);
        CU(cudaMemcpy(&d_ref[i * OFFSET], &d_u1[i][offset], BYTES_PER_GPU, cudaMemcpyDeviceToHost));
    }
    
    for (int i = 0; i < NGPUS; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    cudaSetDevice(0);
    stop_timer(&start, &stop, &milli, "%.4f\n");

    if (DEBUG) {
        print_corners(h_ref, d_ref);
        check_domain_errors(h_ref, d_ref, NX, NY);
        saveResult(d_ref);
    }

    for (int i = 0; i < NGPUS; i++) {
        cudaSetDevice(i);
        CU(cudaFree(d_u1[i]));
        CU(cudaFree(d_u2[i]));
    }

    cudaFreeHost(d_ref);
    free(h_ref);

    cudaDeviceReset();
}
