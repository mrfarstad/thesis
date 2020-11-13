#include "constants.h"
#include "helper_cuda.h"
#include "laplace2d_kernel.cu"

void dispatch_kernels(float *d_u1, float *d_u2) {
    dim3 dimBlock(BLOCK_X,BLOCK_Y);
    dim3 dimGrid(1 + (NX-1)/BLOCK_X, 1 + (NY-1)/BLOCK_Y);
    float *d_tmp;
    for (int i=0; i<ITERATIONS; i++) {
        gpu_laplace2d<<<dimGrid, dimBlock>>>(d_u1, d_u2);
        getLastCudaError("gpu_laplace2d execution failed\n");
        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp; // swap d_u1 and d_u2
    }
}
