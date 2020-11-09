
////////////////////////////////////////////////////////////////////////////////
void Gold_laplace3d(int NX, int NY, float* u1, float* u2) 
{
  int   i, j, ind;
  float fourth=1.0f/4.0f, fifth=1.0f/5.0f;  // predefining this improves performance more than 10%

    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {   // i loop innermost for sequential memory access
	ind = i + j*NX;

        if (i==0 || i==NX-1 || j==0 || j==NY-1) {
          u2[ind] = u1[ind];          // Dirichlet b.c.'s
        }
        else {
            u2[ind] = (
                    //    (u1[ind - 1 + NX]
                    //+   u1[ind + 1 + NX]
                    //+   u1[ind - 1 - NX]
                    //+   u1[ind + 1 - NX]) * fifth 
                    +   u1[ind - 1]
                    +   u1[ind + 1]
                    +   u1[ind - NX]
                    +   u1[ind + NX]) * fourth;
        }
      }
    }
  }

