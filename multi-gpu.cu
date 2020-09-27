#include "stdio.h"
#include "common/common.h"

__global__ void test(int *d_n)
{
    d_n[threadIdx.x]++;
}


int main(int argc, char *argv[])
{

    int ngpus; // 2
    cudaGetDeviceCount(&ngpus);

    for (int i = 0; i < ngpus; i++)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printf("Device %d has compute capability %d.%d.\n", i, devProp.major, devProp.minor); // 7.5
    }

    int hCount = 10;
    int dCount = hCount / ngpus;
    int hSize = hCount * sizeof(int);
    int dSize = hSize / ngpus;

    int *n, *d_n;

    // Allocate host
    n = (int *) malloc(hSize);

    cudaStream_t streams[ngpus];

    // Create streams (one per GPU)
    for (int i = 0; i < ngpus; i++)
    {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < ngpus; i++)
    {
        int idx = i * dCount;
        for (int j = idx; j < idx + dCount; j++)
        {
            n[j] = i;
        }

        // Allocate devices
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((void **) &d_n, dSize));
        CHECK(cudaMemcpyAsync(d_n, &n[idx], dSize, cudaMemcpyHostToDevice, streams[i]));

        test<<<1, dCount, 0, streams[i]>>>(d_n);

        CHECK(cudaMemcpyAsync(&n[idx], d_n, dSize, cudaMemcpyDeviceToHost, streams[i]));
        CHECK(cudaFree(d_n));
    }

    bool error = 0;
    for (int i = 0; i < ngpus; i++)
    {
        int idx = i * dCount;
        for (int j = idx; j < idx + dCount; j++)
        {
            if (n[j] != i + 1)
            {
                printf("Error!\n");
                error = 1;
            } else {
                printf("%d", n[j]);
            }
        }
    }

    if (!error)
    {
        printf("\nSuccess!\n");
    }
    
    // Destroy streams
    for (int i = 0; i < ngpus; i++)
    {
        cudaSetDevice(i);
        cudaStreamDestroy(streams[i]);
    }

    free(n);
}
