#include "stdio.h"
#include "common/common.h"

__global__ void printGPU(int dev)
{
    printf("Hello from GPU %d!\n", dev);
}

void printSmCount()
{
    int device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
}

void printComputeCapabilities(int ngpus)
{
    for (int i = 0; i < ngpus; i++)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printf("Device %d has compute capability %d.%d.\n", i, devProp.major, devProp.minor); // 7.5
    }
}


int main(int argc, char *argv[])
{
    // Fetch number of GPUs
    int ngpus; // 2
    cudaGetDeviceCount(&ngpus);

    // Print GPU properties
    printSmCount();
    printComputeCapabilities(ngpus);

    // Create streams (one per GPU)
    cudaStream_t streams[ngpus];
    for (int i = 0; i < ngpus; i++)
    {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < ngpus; i++)
    {
        cudaSetDevice(i);
        printGPU<<<1, 1>>>(i);
        cudaDeviceSynchronize();
    }
    
    // Destroy streams
    for (int i = 0; i < ngpus; i++)
    {
        cudaSetDevice(i);
        cudaStreamDestroy(streams[i]);
    }
}
