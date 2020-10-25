#include "common.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
// To make a directory
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "cooperative_groups.h"

namespace cg = cooperative_groups;

/*
 * This example implements a 2D stencil computation, spreading the computation
 * across multiple GPUs. This requires communicating halo regions between GPUs
 * on every iteration of the stencil as well as managing multiple GPUs from a
 * single host application. Here, kernels and transfers are issued in
 * breadth-first order to each CUDA stream. Each CUDA stream is associated with
 * a single CUDA device.
 */

#define a0     -3.0124472f
#define a1      1.7383092f
#define a2     -0.2796695f
#define a3      0.0547837f
#define a4     -0.0073118f

// cnst for gpu
#define BDIMX       32
#define NPAD        4
#define NPAD2       8

// constant memories for 8 order FD coefficients
__device__ __constant__ float coef[5];

// set up fd coefficients
void setup_coef (void)
{
    const float h_coef[] = {a0, a1, a2, a3, a4};
    CHECK( cudaMemcpyToSymbol( coef, h_coef, 5 * sizeof(float) ));
}

void saveSnapshotIstep(
    int istep,
    int nx,
    int ny,
    int ngpus,
    float **g_u2)
{
    float *iwave = (float *)malloc(nx * ny * sizeof(float));

    if (ngpus > 1)
    {
        unsigned int skiptop = nx * 4;
        unsigned int gsize = nx * ny / 2;

        for (int i = 0; i < ngpus; i++)
        {
            CHECK(cudaSetDevice(i));
            int iskip = (i == 0 ? 0 : skiptop);
            int ioff  = (i == 0 ? 0 : gsize);
            CHECK(cudaMemcpy(iwave + ioff, g_u2[i] + iskip,
                        gsize * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
    else
    {
        unsigned int isize = nx * ny;
        CHECK(cudaMemcpy (iwave, g_u2[0], isize * sizeof(float),
                          cudaMemcpyDeviceToHost));
    }

    struct stat st = {0};
    // Create snapshots folder if not exists
    if (stat("coop_snapshots", &st) == -1) {
        mkdir("coop_snapshots", 0700);
    }

    char fname[30];
    sprintf(fname, "coop_snapshots/snap_at_step_%d", istep);

    FILE *fp_snap = fopen(fname, "w");

    fwrite(iwave, sizeof(float), nx * ny, fp_snap);
    printf("%s: nx = %d ny = %d istep = %d\n", fname, nx, ny, istep);
    fflush(stdout);
    fclose(fp_snap);

    free(iwave);
    return;
}

/*
 * enable P2P memcopies between GPUs (all GPUs must be compute capability 2.0 or
 * later (Fermi or later))
 */
inline void enableP2P (int ngpus)
{
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

        for (int j = 0; j < ngpus; j++)
        {
            if (i == j) continue;

            int peer_access_available = 0;
            CHECK(cudaDeviceCanAccessPeer(&peer_access_available, i, j));

            if (peer_access_available) CHECK(cudaDeviceEnablePeerAccess(j, 0));
        }
    }
}

inline void calcIndex(int *haloStart, int *haloEnd, int *bodyStart,
                      int *bodyEnd, const int ngpus, const int iny)
{
    // for halo
    for (int i = 0; i < ngpus; i++)
    {
        if (i == 0 && ngpus == 2)
        {
            haloStart[i] = iny - NPAD2;
            haloEnd[i]   = iny - NPAD;

        }
        else
        {
            haloStart[i] = NPAD;
            haloEnd[i]   = NPAD2;
        }
    }

    // for body
    for (int i = 0; i < ngpus; i++)
    {
        if (i == 0 && ngpus == 2)
        {
            bodyStart[i] = NPAD;
            bodyEnd[i]   = iny - NPAD2;
        }
        else
        {
            bodyStart[i] = NPAD + NPAD;
            bodyEnd[i]   = iny - NPAD;
        }
    }
}

inline void calcSkips(int *src_skip, int *dst_skip, const int nx,
                      const int iny)
{
    src_skip[0] = nx * (iny - NPAD2);
    dst_skip[0] = 0;
    src_skip[1] = NPAD * nx;
    dst_skip[1] = (iny - NPAD) * nx;
}

// wavelet
__global__ void kernel_add_wavelet ( float *g_u2, float wavelets, const int nx,
                                     const int ny, const int ngpus)
{
    // global grid idx for (x,y) plane
    int ipos = (ngpus == 2 ? ny - 10 : ny / 2 - 10);
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = ipos * nx + ix;

    if(ix == nx / 2) g_u2[idx] += wavelets;
}

// fd kernel function
__global__ void kernel_2dfd(float *g_u1, float *g_u2, const int nx,
                                 const int iStart, const int iEnd, int ngpus, float **d_u1, float **d_u2)
{
    cg::thread_block g = cg::this_thread_block();

    // global to slice : global grid idx for (x,y) plane
    unsigned int ix  = blockIdx.x * blockDim.x + threadIdx.x;

    // smem idx for current point
    unsigned int stx = threadIdx.x + NPAD;
    unsigned int idx  = ix + iStart * nx;

    // shared memory for u2 with size [4+16+4][4+16+4]
    __shared__ float tile[BDIMX + NPAD2];

    const float alpha = 0.12f;

    // register for y value
    float yval[9];

    for (int i = 0; i < 8; i++) yval[i] = g_u2[idx + (i - 4) * nx];

    // to be used in z loop
    int iskip = NPAD * nx;

#pragma unroll 9
    for (int iy = iStart; iy < iEnd; iy++)
    {
        // get front3 here
        yval[8] = g_u2[idx + iskip];

        if(threadIdx.x < NPAD)
        {
            tile[threadIdx.x]  = g_u2[idx - NPAD];
            tile[stx + BDIMX]    = g_u2[idx + BDIMX];
        }

        tile[stx] = yval[4];
        //__syncthreads();
        g.sync();

        if ( (ix >= NPAD) && (ix < nx - NPAD) )
        {
            // 8rd fd operator
            float tmp = coef[0] * tile[stx] * 2.0f;

#pragma unroll
            for(int d = 1; d <= 4; d++)
            {
                tmp += coef[d] * (tile[stx - d] + tile[stx + d]);
            }

#pragma unroll
            for(int d = 1; d <= 4; d++)
            {
                tmp += coef[d] * (yval[4 - d] + yval[4 + d]);
            }

            // time dimension
            g_u1[idx] = yval[4] + yval[4] - g_u1[idx] + alpha * tmp;
        }

#pragma unroll 8
        for (int i = 0; i < 8 ; i++)
        {
            yval[i] = yval[i + 1];
        }

        // advancd on global idx
        idx  += nx;
        //__syncthreads();
        g.sync();

        // exchange halo
        //if (ngpus > 1)
        //{
        //    // TODO: Sjekk om du får brukt memcpy mellom d_u1[1] og d_u1[0] inne i kernel.
        //    // Jeg er litt usikker på om dette er mulig...
        //    cudaMemcpyAsync(d_u1[1] + dst_skip[0], d_u1[0] + src_skip[0],
        //                iexchange, cudaMemcpyDefault, stream_halo[0]);
        //    cudaMemcpyAsync(d_u1[0] + dst_skip[1], d_u1[1] + src_skip[1],
        //                iexchange, cudaMemcpyDefault, stream_halo[1]);
        //}
    }
}


int main( int argc, char *argv[] )
{
    int ngpus;

    // check device count
    CHECK(cudaGetDeviceCount(&ngpus));
    printf("> CUDA-capable device count: %i\n", ngpus);

    //  get it from command line
    if (argc > 1)
    {
        if (atoi(argv[1]) > ngpus)
        {
            fprintf(stderr, "Invalid number of GPUs specified: %d is greater "
                    "than the total number of GPUs in this platform (%d)\n",
                    atoi(argv[1]), ngpus);
            exit(1);
        }

        ngpus  = atoi(argv[1]);
    }

    printf("> run with device: %i\n", ngpus);

    // size
    const int nsteps  = atoi(argv[2]);
    const int nx      = 512;
    const int ny      = 512;
    const int iny     = ny / ngpus + NPAD * (ngpus - 1);

    size_t isize = nx * iny;
    size_t ibyte = isize * sizeof(float);
    size_t iexchange = NPAD * nx * sizeof(float);

    // set up gpu card
    float *d_u2[ngpus], *d_u1[ngpus];

    for(int i = 0; i < ngpus; i++)
    {
        // set device
        CHECK(cudaSetDevice(i));

        // allocate device memories
        CHECK(cudaMalloc ((void **) &d_u1[i], ibyte));
        CHECK(cudaMalloc ((void **) &d_u2[i], ibyte));

        CHECK(cudaMemset (d_u1[i], 0, ibyte));
        CHECK(cudaMemset (d_u2[i], 0, ibyte));

        printf("GPU %i: allocated %.2f MB gmem\n", i,
               (4.f * ibyte) / (1024.f * 1024.f) );
        setup_coef ();
    }

    // stream definition
    cudaStream_t stream_halo[ngpus], stream_body[ngpus];

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamCreate( &stream_halo[i] ));
        CHECK(cudaStreamCreate( &stream_body[i] ));
    }

    // calculate index for computation
    int haloStart[ngpus], bodyStart[ngpus], haloEnd[ngpus], bodyEnd[ngpus];
    calcIndex(haloStart, haloEnd, bodyStart, bodyEnd, ngpus, iny);

    int src_skip[ngpus], dst_skip[ngpus];

    if(ngpus > 1) calcSkips(src_skip, dst_skip, nx, iny);

    // kernel launch configuration
    dim3 block(BDIMX);
    dim3 grid(nx / block.x);

    // set up event for timing
    CHECK(cudaSetDevice(0));
    cudaEvent_t start, stop;
    CHECK (cudaEventCreate(&start));
    CHECK (cudaEventCreate(&stop ));
    CHECK(cudaEventRecord( start, 0 ));

    // add wavelet only onto gpu0
    CHECK(cudaSetDevice(0));
    kernel_add_wavelet<<<grid, block>>>(d_u2[0], 20.0, nx, iny, ngpus);


    // main loop for wave propagation
    for(int istep = 0; istep < nsteps; istep++)
    {
        // halo part
        for (int i = 0; i < ngpus; i++)
        {
            CHECK(cudaSetDevice(i));

            void *grid_args[] = {
                &d_u1[i],
                &d_u2[i],
                (void *) &nx,
                &haloStart[i],
                &haloEnd[i],
                (void *) &ngpus
            };

            // compute halo
            cudaLaunchCooperativeKernel(
                (void *)kernel_2dfd,
                grid,
                block,
                grid_args,
                0,
                stream_halo[i]
            );

            void *body_args[] = {
                &d_u1[i],
                &d_u2[i],
                (void *) &nx,
                &bodyStart[i],
                &bodyEnd[i],
                (void *) &ngpus
            };

            // compute internal
            cudaLaunchCooperativeKernel(
                (void *)kernel_2dfd,
                grid,
                block,
                body_args,
                0,
                stream_body[i]
            );
        }

        // exchange halo
        if (ngpus > 1)
        {
            // TODO: Sjekk om du får brukt memcpy mellom d_u1[1] og d_u1[0] inne i kernel.
            // Jeg er litt usikker på om dette er mulig...
            CHECK(cudaMemcpyAsync(d_u1[1] + dst_skip[0], d_u1[0] + src_skip[0],
                        iexchange, cudaMemcpyDefault, stream_halo[0]));
            CHECK(cudaMemcpyAsync(d_u1[0] + dst_skip[1], d_u1[1] + src_skip[1],
                        iexchange, cudaMemcpyDefault, stream_halo[1]));
        }

        for (int i = 0; i < ngpus; i++)
        {
            CHECK(cudaSetDevice(i));
            CHECK(cudaDeviceSynchronize());

            float *tmpu0 = d_u1[i];
            d_u1[i] = d_u2[i];
            d_u2[i] = tmpu0;
        }
    }

    // save snap image
    saveSnapshotIstep(nsteps, nx, ny, ngpus, d_u2);

    CHECK(cudaSetDevice( 0 ));
    CHECK(cudaEventRecord( stop, 0 ));

    CHECK(cudaDeviceSynchronize());
    CHECK (cudaGetLastError());

    float elapsed_time_ms = 0.0f;
    CHECK (cudaEventElapsedTime( &elapsed_time_ms, start, stop ));

    //elapsed_time_ms /= nsteps;
    printf("gputime: %8.2fms ", elapsed_time_ms);
    printf("performance: %8.2f MCells/s\n",
           (double) nx * ny / (elapsed_time_ms * 1e3f) );
    fflush(stdout);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // clear
    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

        CHECK (cudaStreamDestroy( stream_halo[i] ));
        CHECK (cudaStreamDestroy( stream_body[i] ));

        CHECK (cudaFree (d_u1[i]));
        CHECK (cudaFree (d_u2[i]));

        CHECK(cudaDeviceReset());
    }

    exit(EXIT_SUCCESS);
}
