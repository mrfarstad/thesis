void initialize_host_region(float *h_u, const int nx, const int ny)
{
    int i, j, idx;
    for (j=0; j<ny; j++) {
        for (i=0; i<nx; i++) {
            idx = i + j*nx;

            if (i==0 || i==nx-1 || j==0 || j==ny-1)
              h_u[idx] = 1.0f;           // Dirichlet b.c.'s
            else
              h_u[idx] = 0.0f;
        }
    }
}
