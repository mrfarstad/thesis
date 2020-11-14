#include "constants.h"
#include "helper_cuda.h"
#include "laplace2d_kernel.cu"


void dispatch_kernels(float **d_u1, float **d_u2, cudaStream_t *streams) {
    dim3 dimBlock(BLOCK_X,BLOCK_Y);
    dim3 dimGrid(1 + (NX-1)/BLOCK_X, 1 + (NY-1)/BLOCK_Y);

    float *d_tmp;
    int i, s, jskip = NY/NGPUS;
    for (i=0; i<ITERATIONS; i++) {
        for (s=0; s<NGPUS; s++) {
        //    //if (SMEM) gpu_laplace2d_smem<<<dimGrid, dimBlock, 0, streams[s]>>>(d_u1, d_u2, start, end);
            gpu_laplace2d_base<<<dimGrid, dimBlock, 0, streams[s]>>>(d_u1[s], d_u2[s]);
            getLastCudaError("gpu_laplace2d execution failed\n");
        }
        //
        // TODO: Fix -> Denne jobber ikke på y: 128 -> 256
        //gpu_laplace2d_base<<<dimGrid, dimBlock, 0, streams[0]>>>(d_u1[0], d_u2[0], 0*jskip, 1*jskip);
        //gpu_laplace2d_base<<<dimGrid, dimBlock, 0, streams[1]>>>(d_u1[1], d_u2[1], 1*jskip, 2*jskip);
        //gpu_laplace2d_base<<<dimGrid, dimBlock>>>(d_u1, d_u2);
        for (s=0; s<NGPUS; s++) {
            cudaStreamSynchronize(streams[s]);
            d_tmp = d_u1[s]; d_u1[s] = d_u2[s]; d_u2[s] = d_tmp; // swap d_u1 and d_u2
        }
        // TODO (multi-gpu): Exchange borders before continuing
    }
}

void dispatch_cooperative_groups_kernels(float *d_u1, float *d_u2) {
    int device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    dim3 dimBlock(BLOCK_X,BLOCK_Y);
    dim3 dimGrid(deviceProp.multiProcessorCount, 1);
    void *args[] = {
        &d_u1,
        &d_u2
    };
    if (SMEM) cudaLaunchCooperativeKernel((void*)gpu_laplace2d_coop_smem, dimGrid, dimBlock, args);
    else cudaLaunchCooperativeKernel((void*)gpu_laplace2d_coop, dimGrid, dimBlock, args);
    getLastCudaError("gpu_laplace2d execution failed\n");
}
