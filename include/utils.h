#ifndef _UTILS_H
#define _UTILS_H

#define PRINT_COMPUTE_CAPABILITIES(ngpus)                                                     \
{                                                                                             \
    for (int i = 0; i < ngpus; i++)                                                           \
    {                                                                                         \
        cudaDeviceProp devProp;                                                               \
        cudaGetDeviceProperties(&devProp, i);                                                 \
        printf("Device %d has compute capability %d.%d \n", i, devProp.major, devProp.minor); \
    }                                                                                         \
}                                                                                             \

#define PRINT_SM_COUNT()                                                                      \
{                                                                                             \
    int device = 0;                                                                           \
    cudaDeviceProp deviceProp;                                                                \
    cudaGetDeviceProperties(&deviceProp, device);                                             \
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);                            \
}                                                                                             \

#define START_TIME(start, stop)                                                               \
{                                                                                             \
    CU(cudaEventCreate(&start));                                                           \
    CU(cudaEventCreate(&stop));                                                            \
    CU(cudaEventRecord(start, 0));                                                         \
}                                                                                             \

#define STOP_TIME(time, start, stop)                                                          \
{                                                                                             \
    /* record stop event on the default stream */                                             \
    CU(cudaEventRecord(stop));                                                             \
    /* wait until the stop event completes */                                                 \
    CU(cudaEventSynchronize(stop));                                                        \
    /* calculate the elapsed time between two events float time */                            \
    CU(cudaEventElapsedTime(&time, start, stop));                                          \
    /* clean up the two events */                                                             \
    CU(cudaEventDestroy(start); cudaEventDestroy(stop));                                   \
}                                                                                             \

#define PRINT_TIME(time)                                                                      \
{                                                                                             \
    printf("The kernel execution took %.4f ms \n", time);                                     \
}                                                                                             \

/*
 * enable P2P memcopies between GPUs (all GPUs must be compute capability 2.0 or
 * later (Fermi or later))
 */                                                                             
#define ENABLE_P2P(ngpus)                                                                     \
{                                                                                             \
    for (int i = 0; i < ngpus; i++)                                                           \
    {                                                                                         \
        CU(cudaSetDevice(i));                                                              \
                                                                                              \
        for (int j = 0; j < ngpus; j++)                                                       \
        {                                                                                     \
            if (i == j) continue;                                                             \
                                                                                              \
            int peer_access_available = 0;                                                    \
            CU(cudaDeviceCanAccessPeer(&peer_access_available, i, j));                     \
                                                                                              \
            if (peer_access_available) CU(cudaDeviceEnablePeerAccess(j, 0));               \
        }                                                                                     \
    }                                                                                         \
}                                                                                             \

#endif // _UTILS_H
