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

/*
 * enable P2P memcopies between GPUs (all GPUs must be compute capability 2.0 or
 * later (Fermi or later))
 */
inline void enableP2P (int ngpus)
{
    for (int i = 0; i < ngpus; i++)
    {
        CU(cudaSetDevice(i));

        for (int j = 0; j < ngpus; j++)
        {
            if (i == j) continue;

            int peer_access_available = 0;
            CU(cudaDeviceCanAccessPeer(&peer_access_available, i, j));

            if (peer_access_available) CU(cudaDeviceEnablePeerAccess(j, 0));
        }
    }
}

int main(int argc, const char **argv){
    float  *h_ref, *d_ref,
           *d_u1[NGPUS], *d_u2[NGPUS];//,
           //milli;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_ref = (float *)malloc(BYTES);
    if (cudaMallocHost((void**)&d_ref, BYTES) != cudaSuccess) {
        fprintf(stderr, "Error returned from pinned host memory allocation\n");
        exit(1);
    }

    //enableP2P(NGPUS);

    print_program_info();

    initialize_host_region(d_ref);

    for (int i = 0; i < NGPUS; i++)
    {
        cudaSetDevice(i);
        CU(cudaMalloc((void **)&d_u1[i], BYTES_PER_GPU + BYTES_BORDER));
        CU(cudaMalloc((void **)&d_u2[i], BYTES_PER_GPU + BYTES_BORDER));
    }

    cudaStream_t streams[NGPUS];

    for (int i = 0; i < NGPUS; i++)
    {
        cudaSetDevice(i);
        CU(cudaStreamCreate( &streams[i] ));
    }

    for (int i = 0; i < NGPUS; i++)
    {
        cudaSetDevice(i);
        CU(cudaMemcpyAsync(&d_u1[i][NX], &d_ref[i * OFFSET], BYTES_PER_GPU, cudaMemcpyHostToDevice, streams[i]));
    }

    readSolution(h_ref);

    //start_timer(start);
    //stop_timer(&start, &stop, &milli, "\nKernel execution time: %.1f (ms) \n");
    //if (COOP) dispatch_cooperative_groups_kernels(d_u1, d_u2);
    //else
    dispatch_kernels(d_u1, d_u2, streams);
    
    for (int i = 0; i < NGPUS; i++)
    {
        cudaSetDevice(i);
        CU(cudaMemcpyAsync(&d_ref[i * OFFSET], &d_u1[i][NX], BYTES_PER_GPU, cudaMemcpyDeviceToHost, streams[i]));
    }
    
    for (int i = 0; i < NGPUS; i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    check_domain_errors(h_ref, d_ref, NX, NY);

    if (DEBUG) print_corners(h_ref, d_ref);
    if (TEST || DEBUG) saveResult(d_ref);

    for (int i = 0; i < NGPUS; i++)
    {
        cudaSetDevice(i);
        CU(cudaStreamDestroy(streams[i]));
        CU(cudaFree(d_u1[i]));
        CU(cudaFree(d_u2[i]));
    }

    cudaFreeHost(d_ref);
    free(h_ref);

    cudaDeviceReset();
}
