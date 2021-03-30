#include <stdio.h>
#include "../include/constants.h"
#include "../include/stencil_initializer.h"
#include "../include/stencil_error_checker.h"
#include "../include/stencil_utils.h"
#include "stencil_cpu.cu"
#include "stencil_dispatch.cu"
#include "omp.h"

int main(int argc, const char **argv) {
    bool error=false;
    float  *h_ref, *d_ref,
           *d_u1[NGPUS], *d_u2[NGPUS],
           milli;

    if (DEBUG) {
        h_ref = (float *)malloc(BYTES);
        char f[] = SOLUTION;
        if (!file_exists(f)) stencil_cpu();
        readSolution(h_ref);
    }

    cudaStream_t streams[NGPUS];
    for (int i = 0; i < NGPUS; i++) {
        cudaSetDevice(i);
        CU(cudaStreamCreate(&streams[i]));
    }

    if (NGPUS>1) ENABLE_P2P(NGPUS);

    if (cudaMallocHost((void**)&d_ref, BYTES) != cudaSuccess) {
        fprintf(stderr, "Error returned from pinned host memory allocation\n");
        exit(1);
    }

    initialize_host_region(d_ref);

    unsigned long size = BYTES_PER_GPU;
    if (NGPUS>1) size += HALO_BYTES;
#pragma omp parallel for num_threads(NGPUS)
    for (int i = 0; i < NGPUS; i++) {
        cudaSetDevice(i);
        CU(cudaMalloc((void **)&d_u1[i], size));
        CU(cudaMalloc((void **)&d_u2[i], size));
    }

    int offset;
    if (NGPUS==1) offset=0;
    else          offset=GHOST_ZONE;
#pragma omp parallel for num_threads(NGPUS)
    for (int i = 0; i < NGPUS; i++) {
        cudaSetDevice(i);
        CU(cudaMemcpyAsync(&d_u1[i][offset], &d_ref[i * OFFSET], BYTES_PER_GPU, cudaMemcpyHostToDevice, streams[i]));
    }

    cudaSetDevice(0);
    cudaEvent_t start, stop;
    CU(cudaEventCreate(&start));
    CU(cudaEventCreate(&stop));
    CU(cudaEventRecord(start));

    if(NGPUS==1) {
        if (COOP) dispatch_cooperative_groups_kernels(d_u1[0], d_u2[0]);
        else      dispatch_kernels(d_u1[0], d_u2[0]);
    } else dispatch_multi_gpu_kernels(d_u1, d_u2, streams);

    cudaSetDevice(0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

#pragma omp parallel for num_threads(NGPUS)
    for (int i = 0; i < NGPUS; i++) {
        cudaSetDevice(i);
        CU(cudaMemcpyAsync(&d_ref[i * OFFSET], &d_u1[i][offset], BYTES_PER_GPU, cudaMemcpyDeviceToHost, streams[i]));
    }
    
    for (int s=0; s<NGPUS; s++) CU(cudaStreamSynchronize(streams[s]));

    if (DEBUG) {
        check_domain_errors(h_ref, d_ref, &error);
        free(h_ref);
    }

    print_program_info();
    printf("%.4f\n", milli); // Print execution time in ms

    CU(cudaFreeHost(d_ref));
    
    for (int i = 0; i < NGPUS; i++) {
        cudaSetDevice(i);
        CU(cudaStreamDestroy(streams[i]));
        CU(cudaFree(d_u1[i]));
        CU(cudaFree(d_u2[i]));
        cudaDeviceReset();
    }

    if (error) exit(EXIT_FAILURE);
    else       exit(EXIT_SUCCESS);
}
