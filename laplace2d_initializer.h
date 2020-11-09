void initialize_host_region(float *h_u)
{
    int i, j, idx;
    for (j=0; j<NY; j++) {
        for (i=0; i<NX; i++) {
            idx = i + j*NX;

            if (i==0 || i==NX-1 || j==0 || j==NY-1)
              h_u[idx] = 1.0f;           // Dirichlet b.c.'s
            else
              h_u[idx] = 0.0f;
        }
    }
}
