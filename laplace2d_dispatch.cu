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

void dispatch_multi_gpu_cooperative_groups_kernels(
        float **d_u1,
        float **d_u2,
        cudaStream_t *streams,
        cudaLaunchParams *launchParams
) {
    int device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    dim3 dimBlock(BLOCK_X,BLOCK_Y);
    dim3 dimGrid(1 + (NX-1)/BLOCK_X, 1 + (NY-1)/BLOCK_Y);
    // TODO: Hvor stort kan et grid være med multi-kernel launch?
    //dim3 dimGrid(deviceProp.multiProcessorCount, 1);
    //void *args[] = {
    //    &d_u1,
    //    &d_u2
    //};
    //if (SMEM) cudaLaunchCooperativeKernel((void*)gpu_laplace2d_coop_smem, dimGrid, dimBlock, args);
    //else cudaLaunchCooperativeKernel((void*)gpu_laplace2d_coop, dimGrid, dimBlock, args);
    //getLastCudaError("gpu_laplace2d execution failed\n");
    
    // TODO: Receive buffers can be used for inter-kernel communication
    //args[i][1] = &r_u[(i+1)%ngpus];
    //args[i][2] = &r_u[i];

    void *args[NGPUS][6];
    int jstarts[NGPUS];
    int jends[NGPUS];
    int jstart = 1;
    int jend = NY/NGPUS;
    int devs[NGPUS];
    for (int s = 0; s < NGPUS; s++)
    {
        args[s][0] = &d_u1[s];
        args[s][1] = &d_u2[s];
        args[s][2] = &d_u1[(s+1)%NGPUS];
        devs[s] = s;
        args[s][3] = (void *)&devs[s];
        //args[s][3] = &d_u1[s];
        if (s==0) {
            jstarts[s] = 1;
            jends[s] = NY/NGPUS+1;
        } else if (s==NGPUS-1) {
            jstarts[s] = 0;
            jends[s] = NY/NGPUS;
        } else {
            jstarts[s] = 0;
            jends[s] = NY/NGPUS+1;
        }
        args[s][4] = &jstarts[s];
        args[s][5] = &jends[s];
        //args[s][2] = (void *)&jstart;
        //args[s][3] = (void *)&jend;
        //if (SMEM) launchParams[s].func = (void*)gpu_laplace2d_coop_smem_multi_gpu;
        //else
        launchParams[s].func = (void*)gpu_laplace2d_coop_multi_gpu;
        launchParams[s].gridDim = dimGrid;
        launchParams[s].blockDim = dimBlock;
        launchParams[s].sharedMem = 0;
        launchParams[s].stream = streams[s];
        launchParams[s].args = args[s];
    }

    cudaLaunchCooperativeKernelMultiDevice(launchParams, NGPUS);
    getLastCudaError("gpu_laplace2d execution failed\n");
}

void dispatch_multi_gpu_kernels(float **d_u1, float **d_u2, cudaStream_t *streams) {
    dim3 dimBlock(BLOCK_X,BLOCK_Y);
    dim3 dimGrid(1 + (NX-1)/BLOCK_X, 1 + (NY-1)/BLOCK_Y);

    float *d_tmp;
    int i, s;
    for (i=0; i<ITERATIONS; i++) {
        for (s=0; s<NGPUS; s++) {
            if (s==0)
                CU(cudaMemcpyAsync(d_u1[s+1], d_u1[s] + (NY/NGPUS) * NX,
                                   NX*sizeof(float), cudaMemcpyDefault, streams[s+1]));
            else if (s==NGPUS-1)
                CU(cudaMemcpyAsync(d_u1[s-1] + (NY/NGPUS + 1) * NX, d_u1[s] + NX,
                                   NX*sizeof(float), cudaMemcpyDefault, streams[s-1]));
            else {
                CU(cudaMemcpyAsync(d_u1[s+1], d_u1[s] + (NY/NGPUS) * NX,
                                   NX*sizeof(float), cudaMemcpyDefault, streams[s+1]));
                CU(cudaMemcpyAsync(d_u1[s-1] + (NY/NGPUS + 1) * NX, d_u1[s] + NX,
                                   NX*sizeof(float), cudaMemcpyDefault, streams[s-1]));
            }
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
            if (SMEM) gpu_laplace2d_smem<<<dimGrid, dimBlock, 0, streams[s]>>>(d_u1[s], d_u2[s], jstart, jend);
            else      gpu_laplace2d_base<<<dimGrid, dimBlock, 0, streams[s]>>>(d_u1[s], d_u2[s], jstart, jend);
            getLastCudaError("gpu_laplace2d execution failed\n");
        }
        
        for (s=0; s<NGPUS; s++) {
            cudaSetDevice(s);
            cudaStreamSynchronize(streams[s]);
            d_tmp = d_u1[s]; d_u1[s] = d_u2[s]; d_u2[s] = d_tmp; // swap d_u1 and d_u2
        }
    }
}
