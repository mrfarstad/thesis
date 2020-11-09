//
// Notes: one thread per node in the 3D block
//

// device code
//

#include "cooperative_groups.h"
using namespace cooperative_groups;

__global__ void gpu_laplace2d(float* __restrict__ d_u1,
			      float* __restrict__ d_u2,
                              int nx,
                              int ny,
                              int iter)
{
    int   i, j, q,
          bx, by,
          gx, gy,
          xskip, yskip, 
          idx, ioff, joff;
    float u2, *d_tmp, fourth=1.0f/4.0f;
    
    //
    // define global indices and array offsets
    //
    //
    bx = blockDim.x;
    by = blockDim.y;

    i  = threadIdx.x + blockIdx.x*bx;
    j  = threadIdx.y + blockIdx.y*by;
    //
    //
    gx = gridDim.x;
    gy = gridDim.y;
    //
    xskip = bx * gx;
    yskip = by * gy;
    
    ioff = 1;
    joff = nx;
    grid_group g = this_grid();
    
    for (q = 1; q <= iter; q++) {
        for (int y=j; y<ny; y+=yskip) {
            for (int x=i; x<nx; x+=xskip) {
                idx = x + y*joff;
        
                if (idx < nx*ny) {
                    if (x==0 || x==nx-1 || y==0 || y==ny-1) {
                      u2 = d_u1[idx]; // Dirichlet b.c.'s
                    }
                    else {
                      u2 = (d_u1[idx-ioff]  +
                            d_u1[idx+ioff]  +
                            d_u1[idx-joff]  +
                            d_u1[idx+joff]) * fourth;    
                    }
                    d_u2[idx] = u2;
                }
            }
        }
        d_tmp = d_u1; d_u1 = d_u2; d_u2 = d_tmp; // swap d_u1 and d_u2
        g.sync();
    }
}
