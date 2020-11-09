#include <stdio.h>
#include "helper_cuda.h"
#include "laplace2d_timer.h"
#include "laplace2d_utils.cu"
#include "laplace2d_kernel.cu"
#include "laplace2d_initializer.h"
#include "laplace2d_error_checker.h"
//#include "cooperative_groups.h"
//using namespace cooperative_groups;

#define CU checkCudaErrors 
#define start_timer cudaEventRecord

#ifndef BLOCK_X
#define BLOCK_X 128
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 4
#endif

#define NX 256
#define NY 256

#define ITERATIONS 8096

void cpu_laplace2d(int nx, int ny, float* h_u1, float* h_u2);

int main(int argc, const char **argv){
    int    i, j, ind,
           ibyte = NX*NY * sizeof(float);
    float  *h_u1, *h_u2, *h_u3, *h_swap,
           *d_u1, *d_u2, *d_tmp,
           milli;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_u1 = (float *)malloc(ibyte);
    h_u2 = (float *)malloc(ibyte);
    h_u3 = (float *)malloc(ibyte);
    CU(cudaMalloc((void **)&d_u1, ibyte));
    CU(cudaMalloc((void **)&d_u2, ibyte));

    initialize_host_region(h_u1);

    start_timer(start);
    CU(cudaMemcpy(d_u1, h_u1, ibyte, cudaMemcpyHostToDevice));
    stop_timer(&start, &stop, &milli, "\ncudaMemcpyHostToDevice: %.1f (ms) \n");

    dim3 dimBlock(BLOCK_X,BLOCK_Y);
    dim3 dimGrid(
        1 + (NX-1)/BLOCK_X,
        1 + (NY-1)/BLOCK_Y
    );

    int nx = NX;
    int ny = NY;
    int iter = ITERATIONS;
    
    void *args[] = {
        &d_u1,
        &d_u2,
        (void *)&nx,
        (void *)&ny,
        (void *)&iter
    };

    int device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    // initialize, then launch
    printf("Multiprocessorcount: %d, Grid: x: %d y: %d\n", deviceProp.multiProcessorCount, dimGrid.x, dimGrid.y);

    start_timer(start);
    cudaLaunchCooperativeKernel((void*)gpu_laplace2d, deviceProp.multiProcessorCount, dimBlock, args);
    stop_timer(&start, &stop, &milli, "\ngpu_laplace2d (cooperative groups): %.1f (ms) \n");
    
    start_timer(start);
    CU(cudaMemcpy(h_u2, d_u1, ibyte, cudaMemcpyDeviceToHost));
    stop_timer(&start, &stop, &milli, "\ncudaMemcpyDeviceToHost: %.1f (ms) \n");


    start_timer(start);
    for (i = 1; i <= ITERATIONS; ++i) {
        cpu_laplace2d(NX, NY, h_u1, h_u3);
        h_swap = h_u1; h_u1 = h_u3; h_u3 = h_swap;   // swap h_u1 and h_u3
    }
    stop_timer(&start, &stop, &milli, "\ncpu_laplace2d: %.1f (ms) \n");

    check_domain_errors(h_u1, h_u2);

    // print out corner of array
    for (j=0; j<8; j++) {
      for (i=0; i<8; i++) {
        ind = i + j*NX;
        printf(" %5.2f ", h_u2[ind]);
      }
      printf("\n");
    }

    saveResult(h_u1);

    CU(cudaFree(d_u1));
    CU(cudaFree(d_u2));
    free(h_u1);
    free(h_u2);
    free(h_u3);

    cudaDeviceReset();
}
