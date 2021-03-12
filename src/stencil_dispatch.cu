#include "../include/constants.h"
#include "../include/helper_cuda.h"
#include "stencil_kernel.cu"

typedef void (*kernel_t)      (float*,float*,unsigned int,unsigned int);
typedef void (*coop_kernel_t) (float*,float*);

kernel_t      get_kernel()      { return SMEM ? gpu_stencil_smem      : gpu_stencil_base; }
coop_kernel_t get_coop_kernel() { return SMEM ? gpu_stencil_coop_smem : gpu_stencil_coop; }

void dispatch_kernels(float *d_u1, float *d_u2) {
    dim3 block(BLOCK_X,BLOCK_Y,BLOCK_Z);
    dim3 grid(1+(NX-1)/BLOCK_X, 1+(NY-1)/BLOCK_Y, 1+(NZ-1)/BLOCK_Z);
    float *d_tmp;
    unsigned int smem = 0;
    if (SMEM) {
        smem = BLOCK_X*BLOCK_Y*BLOCK_Z*sizeof(float);
        //cudaFuncSetAttribute(gpu_stencil_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        // Max on V100: cudaFuncSetAttribute(gpu_stencil_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    }
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

//CHECK(cudaMemcpyAsync(d_u1[1] + dst_skip[0], d_u1[0] + src_skip[0],
//            iexchange, cudaMemcpyDefault, streams_ghost_zone[0]));
//CHECK(cudaMemcpyAsync(d_u1[0] + dst_skip[1], d_u1[1] + src_skip[1],
//            iexchange, cudaMemcpyDefault, streams_ghost_zone[1]));

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

void synchronize_execution(cudaStream_t* streams) {
    for (unsigned int s=0; s<NGPUS; s++) CU(cudaStreamSynchronize(streams[s]));
}

void synchronize_execution() {
    for (unsigned int s=0; s<NGPUS; s++) CU(cudaDeviceSynchronize());
}

void exchange_ghost_zones(float **d_u1, cudaStream_t* streams) {
    int s;
    for (s=0; s<NGPUS-1; s++) send_upper_ghost_zone(d_u1, s, streams);
    for (s=1; s<NGPUS; s++)   send_lower_ghost_zone(d_u1, s, streams);
}

void launch_kernel(kernel_t kernel, dim3 grid, dim3 block, float **d_u1, float **d_u2, cudaStream_t* streams) {
    unsigned int s, kstart, kend;
    for (s=0; s<NGPUS; s++) {
        CU(cudaSetDevice(s));
        kstart = 0;
        kend   = INTERNAL_END-1+HALO_DEPTH;
        if      (s==0)       kstart = INTERNAL_START;
        else if (s==NGPUS-1) kend   = INTERNAL_END-1;
        unsigned int smem = 0;
        if (SMEM) smem = BLOCK_X*BLOCK_Y*BLOCK_Z*sizeof(float);
        kernel<<<grid, block, smem, streams[s]>>>(d_u1[s], d_u2[s], kstart, kend);
        getLastCudaError("kernel execution failed\n");
    }
}

void calculate_ghost_zones(dim3 grid, dim3 block, float **d_u1, float **d_u2, cudaStream_t* streams) {
        if (SMEM) launch_kernel(gpu_stencil_smem_ghost_zone, grid, block, d_u1, d_u2, streams);
        else      launch_kernel(gpu_stencil_base_ghost_zone, grid, block, d_u1, d_u2, streams);
}

void calculate_internal(dim3 grid, dim3 block, float **d_u1, float **d_u2, cudaStream_t* streams) {
        if (SMEM) launch_kernel(gpu_stencil_smem, grid, block, d_u1, d_u2, streams);
        else      launch_kernel(gpu_stencil_base, grid, block, d_u1, d_u2, streams);
}

void dispatch_multi_gpu_kernels(float **d_u1, float **d_u2, cudaStream_t *stream_internal, cudaStream_t *streams_ghost_zone) {
    dim3 block(BLOCK_X,BLOCK_Y,BLOCK_Z);
    dim3 grid(1+(NX-1)/BLOCK_X, 1+(NY-1)/BLOCK_Y, 1+(NZ-1)/BLOCK_Z);
    float **d_tmp;
    unsigned int i;

    exchange_ghost_zones(d_u1, streams_ghost_zone);
    synchronize_execution(streams_ghost_zone);

    for (i=0; i<ITERATIONS; i++) {

        calculate_ghost_zones(grid, block, d_u1, d_u2, streams_ghost_zone);
        synchronize_execution(streams_ghost_zone);

        // Will this be overlapped if you use the same streams? Should we make one internal streams and one ghost zone streams?
        exchange_ghost_zones(d_u2, streams_ghost_zone);
        calculate_internal(grid, block, d_u1, d_u2, stream_internal);

        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp; // swap d_u1 and d_u2

        synchronize_execution();
    }
}
