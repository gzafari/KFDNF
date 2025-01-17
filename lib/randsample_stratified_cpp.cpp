#include "math.h"
#include "mex.h"

#define MEMALLOC(x) mxMalloc(x)
#define MEMFREE(x) mxFree(x)

// THIS FUNCTION randsample_stratified RETURNS A SAMPLE OF NUMBER BETWEEN 
// 0 AND N-1 USING THE WEIGHT SPECIFIED IN w AND THE STRATIFIED SAMPLING 
// METHOD (WITH A RANDOM NUMBER OF u)
//
// INPUT:
//      w:          WEIGHTS (COULD BE UNNORMALIZED).
//      u:          A UNIFORM RANDOM VARIABLE.
//
// OUTPUT:
//      newsample:  INDICES OF NEW SAMPLE.
//
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *newsample, *w, *cum_w;
    int N, i, j; 
    double u, sumw, w_tmp, u_actu;
    
    w = mxGetPr(prhs[0]);
    N = mxGetM(prhs[0]);
    u = mxGetScalar(prhs[1]);
    
    plhs[0] = mxCreateDoubleMatrix(1, N, mxREAL);
    newsample = mxGetPr(plhs[0]);
    
    cum_w = (double*) MEMALLOC(N*sizeof(double));
    
    if (!cum_w )
    {
        if (cum_w) MEMFREE(cum_w);
    }

    sumw = 0;
    for (i = 0; i < N; i++)
    {
        sumw = sumw + w[i];
    }
    
    w_tmp = w[0]/sumw;
    cum_w[0] = w_tmp;
    for (i = 1; i < N; i++)
    {
        w_tmp = w[i]/sumw;
        cum_w[i] = cum_w[i-1] + w_tmp;  
    }
    cum_w[N-1] = 1;
    
    j = 0;
    u_actu = u/((double)N);

    for (i = 0; i < N; i++)
    {
        if (u_actu <= cum_w[j])
        {
            newsample[i] = j+1;
            u_actu = u_actu + 1/((double)N);
        }
        else
        {
            j               = j + 1;
            i               = i - 1;
        }
    } 
    MEMFREE(cum_w);
}
