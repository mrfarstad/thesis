#include "constants.h"
#include "helper_cuda.h"
#include "laplace2d_kernel.cu"

void dispatch_kernels(float *d_u1, float *d_u2) {
    dim3 dimBlock(BLOCK_X,BLOCK_Y);
    dim3 dimGrid(1 + (NX-1)/BLOCK_X, 1 + (NY-1)/BLOCK_Y);
    float *d_tmp;
    for (int i=0; i<ITERATIONS; i++) {
        if (SMEM) gpu_laplace2d_smem<<<dimGrid, dimBlock>>>(d_u1, d_u2, 0, NY-1);
        else      gpu_laplace2d_base<<<dimGrid, dimBlock>>>(d_u1, d_u2, 0, NY-1);
        getLastCudaError("gpu_laplace2d execution failed\n");
        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp; // swap d_u1 and d_u2
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

void dispatch_multi_gpu_kernels(float **d_u1, float **d_u2) {
    dim3 dimBlock(BLOCK_X,BLOCK_Y);
    dim3 dimGrid(1 + (NX-1)/BLOCK_X, 1 + (NY-1)/BLOCK_Y);
    float *d_tmp;
    int i, s;
    for (i=0; i<ITERATIONS; i++) {
        for (s=0; s<NGPUS; s++) {
            cudaSetDevice(s);
            if (s==0)
                CU(cudaMemcpy(d_u1[s] + (NY/NGPUS + 1) * NX, d_u1[s+1] + NX,
                                   NX*sizeof(float), cudaMemcpyDeviceToDevice));
            else if (s==NGPUS-1)
                CU(cudaMemcpy(d_u1[s], d_u1[s-1] + (NY/NGPUS) * NX,
                                   NX*sizeof(float), cudaMemcpyDeviceToDevice));
            else {
                CU(cudaMemcpy(d_u1[s], d_u1[s-1] + (NY/NGPUS) * NX,
                                   NX*sizeof(float), cudaMemcpyDeviceToDevice));
                CU(cudaMemcpy(d_u1[s] + (NY/NGPUS + 1) * NX, d_u1[s+1] + NX,
                                   NX*sizeof(float), cudaMemcpyDeviceToDevice));
            }
            //if (s==0)
            //    CU(cudaMemcpy(d_u1[s+1], d_u1[s] + (NY/NGPUS) * NX,
            //                       NX*sizeof(float), cudaMemcpyDeviceToDevice));
            //else if (s==NGPUS-1)
            //    CU(cudaMemcpy(d_u1[s-1] + (NY/NGPUS + 1) * NX, d_u1[s] + NX,
            //                       NX*sizeof(float), cudaMemcpyDeviceToDevice));
            //else {
            //    CU(cudaMemcpy(d_u1[s+1], d_u1[s] + (NY/NGPUS) * NX,
            //                       NX*sizeof(float), cudaMemcpyDeviceToDevice));
            //    CU(cudaMemcpy(d_u1[s-1] + (NY/NGPUS + 1) * NX, d_u1[s] + NX,
            //                       NX*sizeof(float), cudaMemcpyDeviceToDevice));
            //}
        }
        int jstart, jend;
        for (s=0; s<NGPUS; s++) {
            cudaSetDevice(s);
            if (s==0) {
                jstart = 1;
                jend = NY/NGPUS+1;
            } else if (s==NGPUS-1) {
                jstart = 0;
                jend = NY/NGPUS;
            } else {
                jstart = 0;
                jend = NY/NGPUS+1;
            }
            if (SMEM) gpu_laplace2d_smem<<<dimGrid, dimBlock>>>(d_u1[s], d_u2[s], jstart, jend);
            else      gpu_laplace2d_base<<<dimGrid, dimBlock>>>(d_u1[s], d_u2[s], jstart, jend);
            getLastCudaError("gpu_laplace2d execution failed\n");
        }

        for (s=0; s<NGPUS; s++) {
            cudaSetDevice(s);
            cudaDeviceSynchronize();
            d_tmp = d_u1[s]; d_u1[s] = d_u2[s]; d_u2[s] = d_tmp; // swap d_u1 and d_u2
        }
    }
}
