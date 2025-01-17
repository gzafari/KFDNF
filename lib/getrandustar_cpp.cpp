#include "math.h"
#include "mex.h"

#define MEMALLOC(x) mxMalloc(x)
#define MEMFREE(x) mxFree(x)

// THIS FUNCTION RETURNS r AND ustar TO BE USED IN THE CONTINUOUS 
// RESAMPLING OF MALIK AND PITT (2011)
// 
// INPUT:
//      
//
// OUTPUT:
//      ...
//
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    if(nrhs!=3) {
        mexErrMsgIdAndTxt("MyToolbox:getrandustar_cpp:nrhs", "3 inputs required.");
    }
    
    int i, j, N, PartitionSize;
    double s, u, uc;
    double *w, *r, *ustar;
    
    w = mxGetPr(prhs[0]);
    N = mxGetM(prhs[0]);
    PartitionSize = mxGetScalar(prhs[1]);
    u = mxGetScalar(prhs[2]);
    
    plhs[0] = mxCreateDoubleMatrix(PartitionSize, 1, mxREAL);
    r = mxGetPr(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(PartitionSize, 1, mxREAL);
    ustar = mxGetPr(plhs[1]);
    
    s = 0; j = 0;
    for (i = 0; i < N; i++) {
        s = s + w[i];
        uc = (u + (double)j)/((double)PartitionSize);
        while ( j < PartitionSize && uc <= s) {
            r[j] = (double)(i) + 1.0;
            ustar[j] = (uc - (s -  w[i]))/w[i];
            j = j + 1;
            uc = (u + (double)j)/((double)PartitionSize);
            if ( r[j] == 1 ) {
                ustar[j] = 0;
            }
            if ( r[j] == N ) {
                r[j] = N-1;
                ustar[j] = 1;
            }
        }
    }    
}

