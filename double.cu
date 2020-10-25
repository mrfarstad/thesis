#include "common.h"
#include "utils.h"
#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

__global__
void test(int *g_u1, size_t N, int dev, int *src, int *dest, size_t ibyte)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) g_u1[idx] = (idx + 1) * (dev + 1);

    //memcpy(d_u1[(dev+1) % 2], d_u1[dev], ibyte);
    if (dev == 0) {
        //cudaMemcpyPeerAsync(dest, 1, src, 0, ibyte);
        //cudaMemcpyAsync(src + 1, src, ibyte, cudaMemcpyDeviceToDevice); // works, because src is on the same device
        //cudaMemcpyAsync(dest, src, ibyte, cudaMemcpyDeviceToDevice); // works, because src is on the same device
        //memcpy(dest, src, ibyte);
    }
}

int main(int argc, char *argv[]) {


    int ngpus;
    // get number of devices
    CHECK(cudaGetDeviceCount(&ngpus));
    printf("> CUDA-capable device count: %i\n", ngpus);

    // set up gpu card
    int *d_u1[ngpus];

    size_t isize = 5;
    size_t ibyte = isize * sizeof(int);

    int *host_ref = (int *) malloc(ibyte);

    for (int i = 0; i < ngpus; i++)
    {
        int supportsMdCoopLaunch = 0;
        cudaDeviceGetAttribute(&supportsMdCoopLaunch, cudaDevAttrCooperativeMultiDeviceLaunch, i);
        printf("Device %d supportsMdCoopLaunch? %s\n", i, supportsMdCoopLaunch ? "Yes" : "No");

        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((void **) &d_u1[i], ibyte));
    }

    ENABLE_P2P(ngpus);

    dim3 block(isize);
    dim3 grid(1);

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        test<<<grid, block>>>(d_u1[i], isize, i, d_u1[0], d_u1[1], ibyte);
    }

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemcpy(host_ref + isize * i, d_u1[i], ibyte, cudaMemcpyDeviceToHost));
    }

    size_t ires = ngpus * isize;
    for (int i = 0; i < ires; i++) {
        printf("%d ", host_ref[i]);
    }


    //CHECK(cudaMemcpyAsync(d_u1[1] + dst_skip[0], d_u1[0] + src_skip[0],
    //            iexchange, cudaMemcpyDefault, stream_halo[0]));
    //CHECK(cudaMemcpyAsync(d_u1[0] + dst_skip[1], d_u1[1] + src_skip[1],
    //            iexchange, cudaMemcpyDefault, stream_halo[1]));
    //
    CHECK(cudaDeviceReset());

}
