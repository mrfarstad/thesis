//
// Notes: one thread per node in the 3D block
//

// device code
//

__global__ void GPU_laplace3d(const float* __restrict__ d_u1,
			      float* __restrict__ d_u2,
                              const int blockx,
                              const int blocky,
                              const int blockz,
                              const int nx,
                              const int ny,
                              const int nz)
{
  int   i, j, k, indg, ioff, joff, koff;
  float u2, sixth=1.0f/6.0f;

  //
  // define global indices and array offsets
  //

  i    = threadIdx.x + blockIdx.x*blockx;
  j    = threadIdx.y + blockIdx.y*blocky;
  k    = threadIdx.z + blockIdx.z*blockz;

  ioff = 1;
  joff = nx;
  koff = nx*ny;

  indg = i + j*joff + k*koff;

  if (i>=0 && i<=nx-1 && j>=0 && j<=ny-1 && k>=0 && k<=nz-1) {
    if (i==0 || i==nx-1 || j==0 || j==ny-1 || k==0 || k==nz-1) {
      u2 = d_u1[indg];  // Dirichlet b.c.'s
    }
    else {
      u2 = ( d_u1[indg-ioff] + d_u1[indg+ioff]
           + d_u1[indg-joff] + d_u1[indg+joff]
           + d_u1[indg-koff] + d_u1[indg+koff] ) * sixth;
    }
    d_u2[indg] = u2;
  }
}
