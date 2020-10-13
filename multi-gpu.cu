#include "stdio.h"
#include "common.h"
#include "utils.h"

// Block dimensions
#define BLOCKX 32
#define BLOCKY 16

// Stencil coefficients
#define COEFS 5
#define c0 1.0f
#define c1 1.0f
#define c2 1.0f
#define c3 1.0f
#define c4 1.0f

__global__
void printGPU(int dev)
{
    printf("Hello from GPU %d!\n", dev);
}

__device__ __constant__
float coef[COEFS];

void allocateCoefficients(void)
{
    const float h_coef[] = { c0, c1, c2, c3, c4 };
    CHECK(cudaMemcpyToSymbol(coef, h_coef, COEFS * sizeof(float)))
}

int main(int argc, char *argv[])
{
    // Fetch number of GPUs
    int ngpus; // 2
    cudaGetDeviceCount(&ngpus);

    // Print GPU properties
    PRINT_SM_COUNT();
    PRINT_COMPUTE_CAPABILITIES(ngpus);

    // Allocate constant memory for coefficients
    allocateCoefficients();

    // Create streams (one per GPU)
    cudaStream_t streams[ngpus];
    for (int i = 0; i < ngpus; i++)
    {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
    }

    // Number of iterations
    const int nsteps = 512;

    // Problem size dimensions
    const int nx = 1024;
    const int ny = 1024;

    // Execution dimensions
    dim3 block(BLOCKX, BLOCKY);
    dim3 grid(nx + BLOCKX - 1 / BLOCKX, ny + BLOCKY - 1 / BLOCKY);

    // Start timer for the kernels
    CHECK(cudaSetDevice(0));
    float time;
    cudaEvent_t start, stop;
    //startTime(&start, &stop);
    START_TIME(start, stop);

    // Run kernel on all GPUs
    for (int i = 0; i < ngpus; i++)
    {
        cudaSetDevice(i);
        printGPU<<<1, 1>>>(i);
    }

    // Run kernel on all GPUs
    for (int i = 0; i < ngpus; i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    // Stop timer for kernels
    CHECK(cudaSetDevice(0));
    STOP_TIME(time, start, stop);
    PRINT_TIME(time);

    // Destroy streams
    for (int i = 0; i < ngpus; i++)
    {
        cudaSetDevice(i);
        cudaStreamDestroy(streams[i]);
    }

    // Reset device
    CHECK(cudaDeviceReset());
}
