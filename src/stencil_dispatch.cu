#include "../include/constants.h"
#include "../include/helper_cuda.h"
#include "stencil_kernel_base.cu"
#include "stencil_kernel_smem.cu"
#include "stencil_kernel_coop.cu"

typedef void (*kernel)      (float*,float*,unsigned int,unsigned int);
typedef void (*coop_kernel) (float*,float*);


kernel get_kernel() { 
    if (DIMENSIONS==3) {
        if (SMEM)       return gpu_stencil_smem_3d;
        if (UNROLL_X>1) return gpu_stencil_base_3d_unrolled;
        return gpu_stencil_base_3d;
    } else if (DIMENSIONS==2) {
        if (SMEM) {
            if (REGISTER) return gpu_stencil_smem_2d_register;
            if (UNROLL_X>1) {
                if (PREFETCH) return gpu_stencil_smem_2d_unrolled_prefetch;
                else          return gpu_stencil_smem_2d_unrolled;
            }
            if (PREFETCH) return gpu_stencil_smem_2d_prefetch;
            else          return gpu_stencil_smem_2d;
        }
        if (UNROLL_X>1) return gpu_stencil_base_2d_unrolled;
        else            return gpu_stencil_base_2d;
    } else {
        if (SMEM)       return gpu_stencil_smem_1d;
        if (UNROLL_X>1) return gpu_stencil_base_1d_unrolled;
        return gpu_stencil_base_1d;
    }
}

coop_kernel get_coop_kernel() { return gpu_stencil_coop; }

void set_smem(unsigned int *smem) {
        if (!SMEM)        {*smem = 0; return;}
        else if (PREFETCH) *smem = SMEM_P_X*SMEM_P_Y*BLOCK_Z*sizeof(float);
        else if (REGISTER) *smem = SMEM_P_X*BLOCK_Y*BLOCK_Z*sizeof(float);
        else               *smem = SMEM_X*BLOCK_Y*BLOCK_Z*sizeof(float);
        cudaFuncSetAttribute(get_kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, *smem);
        // Max on V100: cudaFuncSetAttribute(gpu_stencil_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
}

void dispatch_kernels(float *d_u1, float *d_u2) {
    dim3 block(BLOCK_X,BLOCK_Y,BLOCK_Z);
    dim3 grid((1+(NX-1)/BLOCK_X)/UNROLL_X);
    if (DIMENSIONS>1) grid.y = 1+(NY-1)/BLOCK_Y;
    if (DIMENSIONS>2) grid.z = 1+(NZ-1)/BLOCK_Z;
    float *d_tmp;
    unsigned int smem;
    set_smem(&smem);
    for (int i=0; i<ITERATIONS; i++) {
        get_kernel()<<<grid, block, smem>>>(d_u1, d_u2, 0, NZ-1);
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

void send_upper_ghost_zone(float **d_u1, unsigned int dev, cudaStream_t* streams) {
    CU(cudaMemcpyPeerAsync(d_u1[dev+1],
                           dev+1,
                           d_u1[dev] + (INTERNAL_END-HALO_DEPTH) * BORDER_SIZE,
                           dev,
                           GHOST_ZONE_BYTES,
                           streams[dev]));
}

void send_lower_ghost_zone(float **d_u1, unsigned int dev, cudaStream_t* streams) {
    CU(cudaMemcpyPeerAsync(d_u1[dev-1] + INTERNAL_END * BORDER_SIZE,
                           dev-1,
                           d_u1[dev] + INTERNAL_START * BORDER_SIZE,
                           dev,
                           GHOST_ZONE_BYTES,
                           streams[dev]));
}

void dispatch_multi_gpu_kernels(float **d_u1, float **d_u2, cudaStream_t *streams) {
    dim3 block(BLOCK_X,BLOCK_Y,BLOCK_Z);
    dim3 grid((1+(NX-1)/BLOCK_X)/UNROLL_X);
    if      (DIMENSIONS==2) grid.y = 1+(NY/NGPUS+2*HALO_DEPTH-1)/BLOCK_Y;
    else if (DIMENSIONS==3) {
        grid.y = 1+(NY-1)/BLOCK_Y;
        grid.z = 1+(NZ/NGPUS+2*HALO_DEPTH-1)/BLOCK_Z;
    }

    float **d_tmp;
    //int i, s, n, kstart, kend;
    int i, s;
    unsigned int smem, kstart, kend;
    set_smem(&smem);
    //for (i=0; i<ITERATIONS/HALO_DEPTH; i++) {
    for (i=0; i<ITERATIONS; i++) {
        for (s=0; s<NGPUS-1; s++) send_upper_ghost_zone(d_u1, s, streams);
        for (s=1; s<NGPUS; s++)   send_lower_ghost_zone(d_u1, s, streams);
        for (s=0; s<NGPUS; s++)   CU(cudaStreamSynchronize(streams[s]));
        //for (n=0; n<HALO_DEPTH; n++) {
        for (s=0; s<NGPUS; s++) {
            CU(cudaSetDevice(s));
            kstart = 0;
            kend   = INTERNAL_END-1+HALO_DEPTH;
            if      (s==0)       kstart = INTERNAL_START;
            else if (s==NGPUS-1) kend   = INTERNAL_END-1;
            get_kernel()<<<grid, block, smem, streams[s]>>>(d_u1[s], d_u2[s], kstart, kend);
            getLastCudaError("kernel execution failed\n");
        }
        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp; // swap d_u1 and d_u2
        //}
        for (s=0; s<NGPUS; s++) CU(cudaStreamSynchronize(streams[s]));
    }
}
