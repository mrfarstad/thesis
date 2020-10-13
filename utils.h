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
    CHECK(cudaEventCreate(&start));                                                           \
    CHECK(cudaEventCreate(&stop));                                                            \
    CHECK(cudaEventRecord(start, 0));                                                         \
}                                                                                             \

#define STOP_TIME(time, start, stop)                                                          \
{                                                                                             \
    /* record stop event on the default stream */                                             \
    CHECK(cudaEventRecord(stop));                                                             \
    /* wait until the stop event completes */                                                 \
    CHECK(cudaEventSynchronize(stop));                                                        \
    /* calculate the elapsed time between two events float time */                            \
    CHECK(cudaEventElapsedTime(&time, start, stop));                                          \
    /* clean up the two events */                                                             \
    CHECK(cudaEventDestroy(start); cudaEventDestroy(stop));                                   \
}                                                                                             \

#define PRINT_TIME(time)                                                                      \
{                                                                                             \
    printf("The kernel execution took %.4f ms \n", time);                                     \
}                                                                                             \

#endif // _UTILS_H
