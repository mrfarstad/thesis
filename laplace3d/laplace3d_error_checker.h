void check_domain_errors(float *h_u1, float *h_u2)
{
    int   i, j, k, idx;
    float err = 0.0;
    for (k=0; k<NZ; k++) {
        for (j=0; j<NY; j++) {
            for (i=0; i<NX; i++) {
                idx = i + j*NX + k*NX*NY;
                err += (h_u1[idx]-h_u2[idx])*(h_u1[idx]-h_u2[idx]);
            }
        }
    }
    printf("rms error = %f \n",sqrt(err/ (float)(NX*NY*NZ)));
}
