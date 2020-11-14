#include "constants.h"
#include "cooperative_groups.h"
using namespace cooperative_groups;

__global__ void gpu_laplace2d_base(float* __restrict__ d_u1,
			           float* __restrict__ d_u2,
                                   int jstart,
                                   int jend)
{
    int   i, j,
          tx, ty,
          idx, ioff, joff;
    float u2 = 0.0f, fourth=1.0f/4.0f;

    
    tx = threadIdx.x;
    ty = threadIdx.y;
    i  = tx + blockIdx.x*BLOCK_X;
    j  = ty + blockIdx.y*BLOCK_Y;// + jstart;
    ioff = 1;
    joff = NX;

    //if (i==0&&j==0) printf("jstart: %d, jend: %d\n", jstart, jend);

    // Spør du meg, så virker det som at stream 1 gjør all jobben først, så kommer stream to og gjør resten av jobben fra jstart og oppover. Correct!

    idx = i + j *joff;
    if (i>=0 && i<=NX-1 && j>=jstart && j<=jend) {
        if (i==0 || i==NX-1 || j==jstart || j==jend)
          u2 = d_u1[idx]; // Dirichlet b.c.'s
        else {
          u2 = (d_u1[idx-ioff]  +
                d_u1[idx+ioff]  +
                d_u1[idx-joff]  +
                d_u1[idx+joff]) * fourth;
        }
        d_u2[idx] = u2;
    }
}

__global__ void gpu_laplace2d_smem(float* __restrict__ d_u1,
			      float* __restrict__ d_u2)
{
    int   i, j,
          tx, ty, sx, sy,
          idx, ioff, joff;
    float u2 = 0.0f, fourth=1.0f/4.0f;

    tx = threadIdx.x;
    ty = threadIdx.y;
    sx = tx+1;
    sy = ty+1;
    i  = tx + blockIdx.x*BLOCK_X;
    j  = ty + blockIdx.y*BLOCK_Y;
    ioff = 1;
    joff = NX;

    grid_group g = this_grid();
    thread_block tb = this_thread_block();
    __shared__ float smem[BLOCK_Y+2][BLOCK_X+2];

    idx = i + j*joff;
    if (i != 0)           smem[sy][sx-1]   = d_u1[idx-ioff];
    if (i != NX-1)        smem[sy][sx+1]   = d_u1[idx+ioff];
    if (j != 0)           smem[sy-1][sx]   = d_u1[idx-joff];
    if (j != NY-1)        smem[sy+1][sx]   = d_u1[idx+joff];
    smem[sy][sx] = d_u1[idx];
    tb.sync();
    if (i>=0 && i<=NX-1 && j>=0 && j<=NY-1) {
        if (i==0 || i==NX-1 || j==0 || j==NY-1)
          u2 = d_u1[idx]; // Dirichlet b.c.'s
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
          bx, by,
          gx, gy,
          xskip, yskip, 
          idx, ioff, joff;
    float u2, *d_tmp, fourth=1.0f/4.0f;
    
    bx = blockDim.x;
    by = blockDim.y;
    i  = threadIdx.x + blockIdx.x*bx;
    j  = threadIdx.y + blockIdx.y*by;
    gx = gridDim.x;
    gy = gridDim.y;

    xskip = bx * gx;
    yskip = by * gy;
    
    ioff = 1;
    joff = NX;
    grid_group g = this_grid();
    
    for (q = 1; q <= ITERATIONS; q++) {
        for (y=j; y<NY; y+=yskip) {
            for (x=i; x<NX; x+=xskip) {
                idx = x + y*joff;
                if (x==0 || x==NX-1 || y==0 || y==NY-1)
                  u2 = d_u1[idx]; // Dirichlet b.c.'s
                else {
                  u2 = (d_u1[idx-ioff]  +
                        d_u1[idx+ioff]  +
                        d_u1[idx-joff]  +
                        d_u1[idx+joff]) * fourth;    
                }
                d_u2[idx] = u2;
            }
        }
        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp; // swap d_u1 and d_u2
        g.sync();
    }
}

__global__ void gpu_laplace2d_coop_smem(float* __restrict__ d_u1,
			      float* __restrict__ d_u2)
{
    int   i, j, q, x, y,
          tx, ty, sx, sy,
          xskip, yskip, 
          idx, ioff, joff;
    float u2, *d_tmp, fourth=1.0f/4.0f;

    tx = threadIdx.x;
    ty = threadIdx.y;
    sx = tx+1;
    sy = ty+1;
    i  = tx + blockIdx.x*BLOCK_X;
    j  = ty + blockIdx.y*BLOCK_Y;
    xskip = BLOCK_X * gridDim.x;
    yskip = BLOCK_Y * gridDim.y;
    
    ioff = 1;
    joff = NX;
    grid_group g = this_grid();
    thread_block tb = this_thread_block();

    __shared__ float smem[BLOCK_Y+2][BLOCK_X+2];

    for (q = 1; q <= ITERATIONS; q++) {
        for (y=j; y<NY; y+=yskip) {
            for (x=i; x<NX; x+=xskip) {
                idx = x + y*joff;
                tb.sync();
                if (x != 0)           smem[sy][sx-1]   = d_u1[idx-ioff];
                if (x != NX-1)        smem[sy][sx+1]   = d_u1[idx+ioff];
                if (y != 0)           smem[sy-1][sx]   = d_u1[idx-joff];
                if (y != NY-1)        smem[sy+1][sx]   = d_u1[idx+joff];
                smem[sy][sx] = d_u1[idx];
                tb.sync();

                if (x==0 || x==NX-1 || y==0 || y==NY-1) {
                  u2 = d_u1[idx]; // Dirichlet b.c.'s
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
        g.sync();
    }
}
