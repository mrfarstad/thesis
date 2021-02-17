#include "../include/constants.h"
#include "../include/helper_cuda.h"
#include "laplace3d_kernel.cu"

typedef void (*kernel)      (float*,float*,int,int);
typedef void (*coop_kernel) (float*,float*);

kernel      get_kernel()      { return SMEM ? gpu_laplace3d_smem      : gpu_laplace3d_base; }
coop_kernel get_coop_kernel() { return SMEM ? gpu_laplace3d_coop_smem : gpu_laplace3d_coop; }

void dispatch_kernels(float *d_u1, float *d_u2) {
    dim3 block(BLOCK_X,BLOCK_Y,BLOCK_Z);
    dim3 grid(1+(NX-1)/BLOCK_X, 1+(NY-1)/BLOCK_Y, 1+(NZ-1)/BLOCK_Z);
    float *d_tmp;
    for (int i=0; i<ITERATIONS; i++) {
        get_kernel()<<<grid, block>>>(d_u1, d_u2, 0, NY-1);
        getLastCudaError("kernel execution failed\n");
        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp; // swap d_u1 and d_u2
    }
}

void dispatch_cooperative_groups_kernels(float *d_u1, float *d_u2) {
    int device = 0;
    dim3 block(BLOCK_X,BLOCK_Y,BLOCK_Z);
    int numBlocksPerSm = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                  get_coop_kernel(),
                                                  BLOCK_X*BLOCK_Y*BLOCK_Z,
                                                  0);
    dim3 grid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);
    void *args[] = { &d_u1, &d_u2 };
    cudaLaunchCooperativeKernel((void*)get_coop_kernel(),
                                grid,
                                block,
                                args);
    getLastCudaError("kernel execution failed\n");
}

void fetch_upper_ghost_zone(float **d_u1, unsigned int dev, cudaStream_t* streams) {
    CU(cudaMemcpyPeerAsync(d_u1[dev] + (TOP+1) * BORDER_SIZE,
                           dev,
                           d_u1[dev+1] + BOT * BORDER_SIZE,
                           dev+1,
                           GHOST_ZONE_BYTES,
                           streams[dev]));
}

void fetch_lower_ghost_zone(float **d_u1, unsigned int dev, cudaStream_t* streams) {
    CU(cudaMemcpyPeerAsync(d_u1[dev],
                           dev,
                           d_u1[dev-1] + TOP * BORDER_SIZE,
                           dev-1,
                           GHOST_ZONE_BYTES,
                           streams[dev]));
}

void dispatch_multi_gpu_kernels(float **d_u1, float **d_u2, cudaStream_t *streams) {
    dim3 block(BLOCK_X,BLOCK_Y,BLOCK_Z);
    dim3 grid(1+(NX-1)/BLOCK_X, 1+(NY-1)/BLOCK_Y, 1+(NZ-1)/BLOCK_Z);
    float *d_tmp;
    int i, s, n;
    int kstart, kend;
    for (i=0; i<ITERATIONS/HALO_DEPTH; i++) {
        for (s=0; s<NGPUS; s++) {
            cudaSetDevice(s);
            if (s>0)       fetch_lower_ghost_zone(d_u1, s, streams);
            if (s<NGPUS-1) fetch_upper_ghost_zone(d_u1, s, streams);
        }
        for (n = 0; n < HALO_DEPTH; n++) {
            for (s=0; s<NGPUS; s++) {
                cudaSetDevice(s);
                kstart = 0;
                kend   = TOP+HALO_DEPTH;
                if      (s==0)       kstart = BOT;
                else if (s==NGPUS-1) kend   = TOP;
                get_kernel()<<<grid, block, 0, streams[s]>>>(d_u1[s],
                                                             d_u2[s],
                                                             kstart,
                                                             kend);
                getLastCudaError("kernel execution failed\n");
            }
            for (s=0; s<NGPUS; s++) {
                cudaSetDevice(s);
                cudaStreamSynchronize(streams[s]);
                d_tmp = d_u1[s]; d_u1[s] = d_u2[s]; d_u2[s] = d_tmp; // swap d_u1 and d_u2
            }
        }
    }
}
