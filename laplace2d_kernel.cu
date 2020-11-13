//
// Notes: one thread per node in the 3D block
//

// device code
#include "constants.h"

__global__ void gpu_laplace2d(float* __restrict__ d_u1,
			      float* __restrict__ d_u2)
{
    int   i, j,
          tx, ty,
          idx, ioff, joff;
    float u2 = 0.0f, fourth=1.0f/4.0f;
    
    //
    // define global indices and array offsets
    //
    tx = threadIdx.x;
    ty = threadIdx.y;

    i  = tx + blockIdx.x*BLOCK_X;
    j  = ty + blockIdx.y*BLOCK_Y;

    ioff = 1;
    joff = NX;

    idx = i + j*joff;
    if (i>=0 && i<=NX-1 && j>=0 && j<=NY-1) {
        if (i==0 || i==NX-1 || j==0 || j==NY-1)
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
