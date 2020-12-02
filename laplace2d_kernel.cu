#include "constants.h"
#include "cooperative_groups.h"
using namespace cooperative_groups;

__global__ void gpu_laplace2d_base(float* __restrict__ d_u1,
			           float* __restrict__ d_u2,
                                   int jstart,
                                   int jend)
{
    int   i, j, idx;
    float u2 = 0.0f, fourth=1.0f/4.0f;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    idx = i + j *NX;
    if (i>=0 && i<=NX-1 && j>=jstart && j<=jend) {
        if (i==0 || i==NX-1 || j==jstart || j==jend)
          u2 = d_u1[idx]; // Dirichlet boundary conditions
        else {
          u2 = (d_u1[idx-1]   +
                d_u1[idx+1]   +
                d_u1[idx-NX]  +
                d_u1[idx+NX]) * fourth;
        }
        d_u2[idx] = u2;
    }
}

__global__ void gpu_laplace2d_smem(float* __restrict__ d_u1,
			           float* __restrict__ d_u2,
                                   int jstart,
                                   int jend)
{
    int   i, j, sx, sy, idx;
    float u2 = 0.0f, fourth=1.0f/4.0f;
    sx = threadIdx.x+1;
    sy = threadIdx.y+1;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    thread_block tb = this_thread_block();
    __shared__ float smem[BLOCK_Y+2][BLOCK_X+2];
    idx = i + j*NX;
    if (i>=0 && i<=NX-1 && j>=jstart && j<=jend) {
        if (i != 0)      smem[sy][sx-1]   = d_u1[idx-1];
        if (i != NX-1)   smem[sy][sx+1]   = d_u1[idx+1];
        if (j != jstart) smem[sy-1][sx]   = d_u1[idx-NX];
        if (j != jend)   smem[sy+1][sx]   = d_u1[idx+NX];
        smem[sy][sx] = d_u1[idx];
    }
    tb.sync();
    if (i>=0 && i<=NX-1 && j>=jstart && j<=jend) {
        if (i==0 || i==NX-1 || j==jstart || j==jend)
          u2 = d_u1[idx]; // Dirichlet boundary conditions
        else {
          u2 = (smem[sy][sx-1]  +
                smem[sy][sx+1]  +
                smem[sy-1][sx]  +
                smem[sy+1][sx]) * fourth;
        }
        d_u2[idx] = u2;
    }
}

__global__ void gpu_laplace2d_coop(float* __restrict__ d_u1,
			      float* __restrict__ d_u2)
{
    int   i, j, q, x, y,
          xskip, yskip, 
          idx;
    float u2, *d_tmp, fourth=1.0f/4.0f;
    i  = threadIdx.x + blockIdx.x*blockDim.x;
    j  = threadIdx.y + blockIdx.y*blockDim.y;
    xskip = blockDim.x * gridDim.x;
    yskip = blockDim.y * gridDim.y;
    grid_group g = this_grid();
    for (q = 1; q <= ITERATIONS; q++) {
        for (y=j; y<NY; y+=yskip) {
            for (x=i; x<NX; x+=xskip) {
                idx = x + y*NX;
                if (x==0 || x==NX-1 || y==0 || y==NY-1)
                  u2 = d_u1[idx]; // Dirichlet boundary conditions
                else {
                  u2 = (d_u1[idx-1]   +
                        d_u1[idx+1]   +
                        d_u1[idx-NX]  +
                        d_u1[idx+NX]) * fourth;    
                }
                d_u2[idx] = u2;
            }
        }
        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp; // swap d_u1 and d_u2
        g.sync(); // inter-block sync within grid
    }
}

__global__ void gpu_laplace2d_coop_smem(float* __restrict__ d_u1,
			      float* __restrict__ d_u2)
{
    int   i, j, q, x, y, sx, sy, xskip, yskip, idx;
    float u2, *d_tmp, fourth=1.0f/4.0f;
    sx = threadIdx.x+1;
    sy = threadIdx.y+1;
    i  = threadIdx.x + blockIdx.x*BLOCK_X;
    j  = threadIdx.y + blockIdx.y*BLOCK_Y;
    xskip = BLOCK_X * gridDim.x;
    yskip = BLOCK_Y * gridDim.y;
    grid_group g = this_grid();
    thread_block tb = this_thread_block();
    __shared__ float smem[BLOCK_Y+2][BLOCK_X+2];
    for (q = 1; q <= ITERATIONS; q++) {
        for (y=j; y<NY; y+=yskip) {
            for (x=i; x<NX; x+=xskip) {
                idx = x + y*NX;
                tb.sync();
                if (x != 0)    smem[sy][sx-1]   = d_u1[idx-1];
                if (x != NX-1) smem[sy][sx+1]   = d_u1[idx+1];
                if (y != 0)    smem[sy-1][sx]   = d_u1[idx-NX];
                if (y != NY-1) smem[sy+1][sx]   = d_u1[idx+NX];
                smem[sy][sx] = d_u1[idx];
                tb.sync();
                if (x==0 || x==NX-1 || y==0 || y==NY-1) {
                  u2 = d_u1[idx]; // Dirichlet boundary conditions
                }
                else {
                  u2 = (smem[sy][sx-1]  +
                        smem[sy][sx+1]  +
                        smem[sy-1][sx]  +
                        smem[sy+1][sx]) * fourth;
                }
                d_u2[idx] = u2;
            }
        }
        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp; // swap d_u1 and d_u2
        g.sync(); // inter-block sync within grid
    }
}
