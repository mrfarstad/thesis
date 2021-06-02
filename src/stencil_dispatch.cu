#include "../include/constants.h"
#include "../include/helper_cuda.h"
#include "../include/stencil_utils.h"
#include "stencil_kernel_base.cu"
#include "stencil_kernel_smem.cu"
#include "stencil_kernel_smem_padded.cu"
#include "stencil_kernel_smem_register.cu"
#include "stencil_kernel_coop.cu"

typedef void (*kernel)      (float*,float*,unsigned int,unsigned int);
typedef void (*coop_kernel) (float*,float*);


kernel get_kernel() { 
    if (DIMENSIONS==3) {
        if (SMEM) {
            if (UNROLL_X>1) {
                if (PADDED) return smem_padded_unroll_3d;
                if (REGISTER) return smem_register_unroll_3d;
                return smem_unroll_3d;
            }
            if (PADDED) return smem_padded_3d;
            if (REGISTER) return smem_register_3d;
            return smem_3d;
        }
        if (UNROLL_X>1) return base_unroll_3d;
        return base_3d;
    } else {
        if (SMEM) {
            if (UNROLL_X>1) {
                if (PADDED) return smem_padded_unroll_2d;
                if (REGISTER) return smem_register_unroll_2d;
                return smem_unroll_2d;
            }
            if (PADDED) return smem_padded_2d;
            if (REGISTER) return smem_register_2d;
            return smem_2d;
        }
        if (UNROLL_X>1) return base_unroll_2d;
        return base_2d;
    }
}

coop_kernel get_coop_kernel() { return coop; }

void set_smem(unsigned int *smem, unsigned int bx, unsigned int by, unsigned int bz) {
    if (!SMEM) {*smem = 0; return;}
    unsigned int smem_x   = bx*UNROLL_X;
    unsigned int smem_p_x = smem_x + 2*STENCIL_DEPTH;
    unsigned int smem_p_y = by + 2*STENCIL_DEPTH;
    unsigned int smem_p_z = bz + 2*STENCIL_DEPTH;
    if (DIMENSIONS == 3) {
        if (PADDED)        *smem = smem_p_x*smem_p_y*smem_p_z*sizeof(float);
        else if (REGISTER) *smem = smem_p_x*smem_p_y*bz*sizeof(float);
        else               *smem = smem_x*by*bz*sizeof(float);
    } else {
        if (PADDED)        *smem = smem_p_x*smem_p_y*sizeof(float);
        else if (REGISTER) *smem = smem_p_x*by*sizeof(float);
        else               *smem = smem_x*by*sizeof(float);
    }
}

int comp (const void * elem1, const void * elem2) 
{
    int f = *((int*)elem1);
    int s = *((int*)elem2);
    if (f > s) return -1;
    if (f < s) return 1;
    return 0;
}

__host__ __device__ void swap(int *x, int*y) {
    int tmp = *x;
    *x = *y;
    *y = tmp;
}

__host__ __device__ void sort3_desc(int *b0, int *b1, int *b2) {
    if (*b0 < *b2) swap(b0, b2);
    if (*b0 < *b1) swap(b0, b1);
    if (*b1 < *b2) swap(b1, b2);
}

__host__ __device__ void find_3d_block_dimensions(int *bx, int *by, int *bz, int b) {
        int b0 = BLOCK_X;
        while (SMEM && PADDED && b / (b0*STENCIL_DEPTH*STENCIL_DEPTH) == 0 && b0 > 1)
            b0 = b0/2;
        int b1 = MIN(MAX(2, STENCIL_DEPTH), 8);
        int b2 = b/(b0*b1);
        sort3_desc(&b0, &b1, &b2);
        *bx = b0, *by = b1, *bz = b2;
}

__host__ __device__ void set_max_occupancy_block_dimensions(int *bx, int *by, int *bz, int threads) {
        if (DIMENSIONS==3) find_3d_block_dimensions(bx,by,bz,threads);
        else *bx = BLOCK_X, *by = threads/BLOCK_X, *bz=1;
}

struct calculate_smem: std::unary_function<int, int> {
    __host__ __device__ int operator()(int threads) const {
        if (!SMEM) return 0;
        int bx, by, bz;
        set_max_occupancy_block_dimensions(&bx, &by, &bz, threads);
        unsigned int smem_x   = bx*UNROLL_X;
        unsigned int smem_p_x = smem_x + 2*STENCIL_DEPTH;
        unsigned int smem_p_y = by + 2*STENCIL_DEPTH;
        unsigned int smem_p_z = bz + 2*STENCIL_DEPTH;
        unsigned int smem;
        if (DIMENSIONS == 3) {
            if (PADDED)        smem = smem_p_x*smem_p_y*smem_p_z*sizeof(float);
            else if (REGISTER) smem = smem_p_x*smem_p_y*bz*sizeof(float);
            else               smem = smem_x*by*bz*sizeof(float);
        } else {
            if (PADDED)        smem = smem_p_x*smem_p_y*sizeof(float);
            else if (REGISTER) smem = smem_p_x*by*sizeof(float);
            else               smem = smem_x*by*sizeof(float);
        }
        return smem;
    }
};

void set_block_dims(int *bx, int *by, int *bz, int threads) {
    if (HEURISTIC) {
        set_max_occupancy_block_dimensions(bx, by, bz, threads);
    } else {
        *bx = BLOCK_X;
        *by = BLOCK_Y;
        *bz = BLOCK_Z;
    }
}

void dispatch_kernels(float *d_u1, float *d_u2) {
    calculate_smem calc_smem;
    int g, b, bx, by, bz;
    unsigned int smem;
    if (SMEM) {
        const char* arch = STR(ARCH);
        if (strcmp(arch, "volta")==0)
            cudaFuncSetAttribute(get_kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
        else if (strcmp(arch, "pascal")==0)
            cudaFuncSetAttribute(get_kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
    }
    if (HEURISTIC) cudaOccupancyMaxPotentialBlockSizeVariableSMem(&g, &b, get_kernel(), calc_smem);
    set_block_dims(&bx, &by, &bz, b);
    check_early_exit(bx, by, bz);
    print_program_info(bx, by, bz);
    dim3 block(bx, by, bz);
    dim3 grid((1+(NX-1)/bx)/UNROLL_X);
    if (DIMENSIONS>1) grid.y = 1+(NY-1)/by;
    if (DIMENSIONS>2) grid.z = 1+(NZ-1)/bz;
    float *d_tmp;
    set_smem(&smem, bx, by, bz);
    for (int i=0; i<ITERATIONS; i++) {
        get_kernel()<<<grid, block, smem>>>(d_u1, d_u2, 0, NZ-1);
        getLastCudaError("kernel execution failed\n");
        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp; // swap d_u1 and d_u2
    }
}

void dispatch_cooperative_groups_kernels(float *d_u1, float *d_u2) {
//    int device = 0;
//    dim3 block(BLOCK_X,BLOCK_Y,BLOCK_Z);
//    int numBlocksPerSm = 0;
//    cudaDeviceProp deviceProp;
//    cudaGetDeviceProperties(&deviceProp, device);
//    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
//                                                  get_coop_kernel(),
//                                                  BLOCK_X*BLOCK_Y*BLOCK_Z,
//                                                  0);
//    dim3 grid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);
//    void *args[] = { &d_u1, &d_u2 };
//    cudaLaunchCooperativeKernel((void*)get_coop_kernel(),
//                                grid,
//                                block,
//                                args);
//    getLastCudaError("kernel execution failed\n");
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
    calculate_smem calc_smem;
    float **d_tmp;
    int g, b, bx, by, bz, s;
    unsigned int i, kstart, kend, smem;
    if (SMEM) cudaFuncSetAttribute(get_kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    if (HEURISTIC) cudaOccupancyMaxPotentialBlockSizeVariableSMem(&g, &b, get_kernel(), calc_smem);
    set_block_dims(&bx, &by, &bz, b);
    print_program_info(bx, by, bz);
    dim3 block(bx, by, bz);
    dim3 grid((1+(NX-1)/bx)/UNROLL_X);
    if (DIMENSIONS==2) grid.y = 1+(NY/NGPUS+2*HALO_DEPTH-1)/by;
    else if (DIMENSIONS==3) {
        grid.y = 1+(NY-1)/by;
        grid.z = 1+(NZ/NGPUS+2*HALO_DEPTH-1)/bz;
    }
    set_smem(&smem, bx, by, bz);
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
