#include "../include/constants.h"
#include "../include/helper_cuda.h"
#include "laplace3d_kernel.cu"

void dispatch_kernels(float *d_u1, float *d_u2) {
    dim3 block(BLOCK_X,BLOCK_Y,BLOCK_Z);
    dim3 grid(1 + (NX-1)/BLOCK_X, 1 + (NY-1)/BLOCK_Y, 1 + (NZ-1)/BLOCK_Z);
    float *d_tmp;
    for (int i=0; i<ITERATIONS/SMEM_HALO_DEPTH; i++) {
        if (SMEM) gpu_laplace3d_smem<<<grid, block>>>(d_u1, d_u2, 0, NY-1);
        else      gpu_laplace3d_base<<<grid, block>>>(d_u1, d_u2, 0, NY-1);
        getLastCudaError("gpu_laplace3d execution failed\n");
        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp; // swap d_u1 and d_u2
    }
}

void dispatch_cooperative_groups_kernels(float *d_u1, float *d_u2) {
    int device = 0;
    dim3 block(BLOCK_X,BLOCK_Y,BLOCK_Z);
    int numBlocksPerSm = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    if (SMEM)
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                      (void*)gpu_laplace3d_coop_smem,
                                                      BLOCK_X*BLOCK_Y*BLOCK_Z,
                                                      0);
    else
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                      (void*)gpu_laplace3d_coop,
                                                      BLOCK_X*BLOCK_Y*BLOCK_Z,
                                                      0);
    dim3 grid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);
    void *args[] = {
        &d_u1,
        &d_u2
    };
    if (SMEM)
        cudaLaunchCooperativeKernel((void*)gpu_laplace3d_coop_smem,
                                    grid,
                                    block,
                                    args);
    else
        cudaLaunchCooperativeKernel((void*)gpu_laplace3d_coop,
                                    grid,
                                    block,
                                    args);
    getLastCudaError("gpu_laplace3d execution failed\n");
}

void dispatch_multi_gpu_kernels(float **d_u1, float **d_u2, cudaStream_t *streams) {
    dim3 block(BLOCK_X,BLOCK_Y);
    dim3 grid(1 + (NX-1)/BLOCK_X, 1 + (NY-1)/BLOCK_Y);
    float *d_tmp;
    int i, s, n;
    int jstart, jend;

    int bot = HALO_DEPTH;
    int top = HALO_DEPTH+NY/NGPUS-1;

    for (i=0; i<ITERATIONS/HALO_DEPTH; i++) {
        for (s=0; s<NGPUS; s++) {
            cudaSetDevice(s);
            if (s==0)
                CU(cudaMemcpyPeerAsync(d_u1[s] + (top+1) * NX,
                                       s,
                                       d_u1[s+1] + bot * NX,
                                       s+1,
                                       BORDER_BYTES,
                                       streams[s]));
            else if (s==NGPUS-1)
                CU(cudaMemcpyPeerAsync(d_u1[s],
                                       s,
                                       d_u1[s-1] + top * NX,
                                       s-1,
                                       BORDER_BYTES,
                                       streams[s]));
            else {
                CU(cudaMemcpyPeerAsync(d_u1[s],
                                       s,
                                       d_u1[s-1] + top * NX,
                                       s-1,
                                       BORDER_BYTES,
                                       streams[s]));
                CU(cudaMemcpyPeerAsync(d_u1[s] + (top+1) * NX,
                                       s,
                                       d_u1[s+1] + bot * NX,
                                       s+1,
                                       BORDER_BYTES,
                                       streams[s]));
            }
        }
        for (n = 0; n < HALO_DEPTH; n++) {
            for (s=0; s<NGPUS; s++) {
                cudaSetDevice(s);
                jstart = bot;
                jend = top;
                if (s==0) {
                    jstart = bot;
                    jend = top+HALO_DEPTH;
                } else if (s==NGPUS-1) {
                    jstart = 0;
                    jend = top;
                } else {
                    jstart = 0;
                    jend = top+HALO_DEPTH;
                }
                if (SMEM)
                    gpu_laplace3d_smem<<<grid, block, 0, streams[s]>>>(d_u1[s],
                                                                       d_u2[s],
                                                                       jstart,
                                                                       jend);
                else
                    gpu_laplace3d_base<<<grid, block, 0, streams[s]>>>(d_u1[s],
                                                                       d_u2[s],
                                                                       jstart,
                                                                       jend);
                getLastCudaError("gpu_laplace3d execution failed\n");
            }
            for (s=0; s<NGPUS; s++) {
                cudaSetDevice(s);
                cudaStreamSynchronize(streams[s]);
                d_tmp = d_u1[s]; d_u1[s] = d_u2[s]; d_u2[s] = d_tmp; // swap d_u1 and d_u2
            }
        }
    }
}
