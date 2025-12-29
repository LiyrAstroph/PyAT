
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "libccf.h"
#include "gsl_rng.h"

#define EPS (1.0e-20)

/* c function for calculating ICCF */
void ciccf(double t1[], double f1[], int n1, double t2[], double f2[], int n2,
    int ntau, double tau_beg, double tau_end, double threshold, char mode, int ignore_warning,
    double tau[], double ccf[], double *rmax, double *tau_peak, double *tau_cent)
{
  int i, j, jbeg, jend, nintp, nmax, nabove, idx_max;
  double dtau, taui, mean1, mean2, std1, std2, std12, ccf12, w12, ccf21, w21;
  double ccf_max, tau_temp, norm_temp, dtau_above_min;
  double *fintp;
  int *idx_above;

  /* initialize tau and ccf */
  dtau = (tau_end-tau_beg)/(ntau-1);
  for(i=0; i<ntau; i++)
  {
    tau[i] = tau_beg + dtau * i;
    ccf[i] = 0.0;
  }
  
  nmax = (n1>n2?n1:n2);
  fintp = (double *)malloc(nmax*sizeof(double));

  for(i=0; i<ntau; i++)
  {
    taui = tau[i];

    /* first interpolate f2 */
    jbeg = locate_upp(t1, n1, t2[0]   -taui);
    jend = locate_low(t1, n1, t2[n2-1]-taui);
    nintp = jend - jbeg + 1;
    
    ccf12 = 0.0;
    w12 = 0.0;
    if(nintp > 1)
    {
      for(j=jbeg; j<=jend; j++)
      {
        fintp[j] = interpolate(t2, f2, n2, t1[j]+taui);
      }
      
      mean1 = mean_cal(f1+jbeg,    nintp);
      mean2 = mean_cal(fintp+jbeg, nintp);
      std1  = std_cal(f1+jbeg,     nintp);
      std2  = std_cal(fintp+jbeg,  nintp);
      
      std12 = std1*std2;
      if(std12 > 0.0)
      {
        ccf12 = 0.0;
        for(j=jbeg; j<=jend; j++)
        {
          ccf12 += (f1[j]-mean1)*(fintp[j]-mean2);
        }
        ccf12 = ccf12/nintp/std12;
        w12 = 1.0;
      }
      else 
      {
        ccf12 = 0.0;
        w12 = 0.0;
      }
    }
    
    /* then interpolate f1 */
    jbeg = locate_upp(t2, n2, t1[0]   +taui);
    jend = locate_low(t2, n2, t1[n1-1]+taui);
    nintp = jend - jbeg + 1;
    
    ccf21 = 0.0;
    w21 = 0.0;
    if(nintp > 1)
    {
      for(j=jbeg; j<=jend; j++)
      {
        fintp[j] = interpolate(t1, f1, n1, t2[j]-taui);
      }
      
      mean1 = mean_cal(fintp+jbeg, nintp);
      mean2 = mean_cal(f2+jbeg,    nintp);
      std1  = std_cal(fintp+jbeg,  nintp);
      std2  = std_cal(f2+jbeg,     nintp);
      
      std12 = std1*std2;
      if(std12 > 0.0)
      {
        ccf21 = 0.0;
        for(j=jbeg; j<=jend; j++)
        {
          ccf21 += (fintp[j]-mean1)*(f2[j]-mean2);
        }
        ccf21 = ccf21/nintp/std12;
        w21 = 1.0;
      }
      else 
      {
        ccf21 = 0.0;
        w21 = 0.0;
      }
    }

    ccf[i] = (ccf12*w12 + ccf21*w21)/(w12+w21 + EPS);
  }
  
  /* determine the CCF peak */
  ccf_max = -1.0;
  for(i=0; i<ntau; i++)
  {
    if(ccf[i]>ccf_max) ccf_max = ccf[i];
  }
  *rmax = ccf_max;

  /* determine the peak tau using the rightmost one */
  tau_temp = tau[ntau-1];
  for(i=ntau-1; i>=0; i--)
  {
    if(ccf[i]==ccf_max)
    {
      tau_temp = tau[i];
      idx_max = i;
      break;
    }
  }
  *tau_peak = tau_temp;
  
  /* determine the centroid tau */
  /* first search the elements above rmax*threshold */
  idx_above = (unsigned int *)malloc(ntau*sizeof(unsigned int));
  nabove=0;
  for(i=0; i<ntau; i++)
  {
    if(ccf[i] >= ccf_max*threshold)idx_above[nabove++] = i;
  }

  if(ignore_warning == 0)
  {
    if(idx_above[0] == 0)
    {
      printf("tau_beg is too large to cover the region with ccf>threshold*ccf_peak.\n");
    }
    if(idx_above[nabove-1] == ntau-1)
    {
      printf("tau_end is too small to cover the region with ccf>threshold*ccf_peak.\n");
    }
  }

  if(nabove == 1)  /* only one point above */
  {
    *tau_cent = *tau_peak;
  }
  else
  {
    if(mode == 'm')  /* centriod of multiple peaks */
    {
      tau_temp = 0.0;
      norm_temp = 0.0;
      for(i=0; i<nabove; i++)
      {
        j = idx_above[i];
        tau_temp  += ccf[j] * tau[j];
        norm_temp += ccf[j];
      }
      tau_temp = tau_temp/(norm_temp+EPS);
    }
    else /* centroid of the major single peak */
    {
      int idx_above_max, idx_left, idx_right;
      /* starting from the index corresponding to rmax */
      idx_above_max = locate_int_low(idx_above, nabove, idx_max);

      /* go right */
      idx_right = nabove-1;
      for(i=idx_above_max+1; i<nabove; i++)
      {
        if(idx_above[i]-idx_above[i-1] > 1) 
        {
          idx_right = i-1;
          break;
        }
      }
      /* go left */
      idx_left = 0;
      for(i=idx_above_max-1; i>=0; i--)
      {
        if(idx_above[i+1]-idx_above[i] > 1) 
        {
          idx_left = i+1;
          break;
        }
      }
      tau_temp = 0.0;
      norm_temp = 0.0;
      for(i=idx_left; i<=idx_right; i++)
      {
        tau_temp  += ccf[idx_above[i]]*tau[idx_above[i]];
        norm_temp += ccf[idx_above[i]];
      }
      tau_temp = tau_temp/(norm_temp+EPS);
    }

    *tau_cent = tau_temp;
  }

  free(fintp);
  free(idx_above);
  return;
}

/* c function for calculating ICCF */
void ciccf_oneway(double t1[], double f1[], int n1, double t2[], double f2[], int n2,
    int ntau, double tau_beg, double tau_end, double threshold, char mode, int ignore_warning,
    double tau[], double ccf[], double *rmax, double *tau_peak, double *tau_cent)
{
  int i, j, jbeg, jend, nintp, nmax, nabove, idx_max;
  double dtau, taui, mean1, mean2, std1, std2, std12, ccf12, w12, ccf21, w21;
  double ccf_max, tau_temp, norm_temp, dtau_above_min;
  double *fintp;
  int *idx_above;

  /* initialize tau and ccf */
  dtau = (tau_end-tau_beg)/(ntau-1);
  for(i=0; i<ntau; i++)
  {
    tau[i] = tau_beg + dtau * i;
    ccf[i] = 0.0;
  }
  
  nmax = (n1>n2?n1:n2);
  fintp = (double *)malloc(nmax*sizeof(double));

  for(i=0; i<ntau; i++)
  {
    taui = tau[i];

    /* interpolate f1 */
    jbeg = locate_upp(t2, n2, t1[0]   +taui);
    jend = locate_low(t2, n2, t1[n1-1]+taui);
    nintp = jend - jbeg + 1;
    
    ccf21 = 0.0;
    w21 = 0.0;
    if(nintp > 1)
    {
      for(j=jbeg; j<=jend; j++)
      {
        fintp[j] = interpolate(t1, f1, n1, t2[j]-taui);
      }
      
      mean1 = mean_cal(fintp+jbeg, nintp);
      mean2 = mean_cal(f2+jbeg,    nintp);
      std1  = std_cal(fintp+jbeg,  nintp);
      std2  = std_cal(f2+jbeg,     nintp);
      
      std12 = std1*std2;
      if(std12 > 0.0)
      {
        ccf21 = 0.0;
        for(j=jbeg; j<=jend; j++)
        {
          ccf21 += (fintp[j]-mean1)*(f2[j]-mean2);
        }
        ccf21 = ccf21/nintp/std12;
        w21 = 1.0;
      }
      else 
      {
        ccf21 = 0.0;
        w21 = 0.0;
      }
    }

    ccf[i] = ccf21;
  }
  
  /* determine the CCF peak */
  ccf_max = -1.0;
  for(i=0; i<ntau; i++)
  {
    if(ccf[i]>ccf_max) ccf_max = ccf[i];
  }
  *rmax = ccf_max;

  /* determine the peak tau using the rightmost one */
  tau_temp = tau[ntau-1];
  for(i=ntau-1; i>=0; i--)
  {
    if(ccf[i]==ccf_max)
    {
      tau_temp = tau[i];
      idx_max = i;
      break;
    }
  }
  *tau_peak = tau_temp;
  
  /* determine the centroid tau */
  /* first search the elements above rmax*threshold */
  idx_above = (unsigned int *)malloc(ntau*sizeof(unsigned int));
  nabove=0;
  for(i=0; i<ntau; i++)
  {
    if(ccf[i] >= ccf_max*threshold)idx_above[nabove++] = i;
  }

  if(ignore_warning == 0)
  {
    if(idx_above[0] == 0)
    {
      printf("tau_beg is too large to cover the region with ccf>threshold*ccf_peak.\n");
    }
    if(idx_above[nabove-1] == ntau-1)
    {
      printf("tau_end is too small to cover the region with ccf>threshold*ccf_peak.\n");
    }
  }

  if(nabove == 1)  /* only one point above */
  {
    *tau_cent = *tau_peak;
  }
  else
  {
    if(mode == 'm')  /* centriod of multiple peaks */
    {
      tau_temp = 0.0;
      norm_temp = 0.0;
      for(i=0; i<nabove; i++)
      {
        j = idx_above[i];
        tau_temp  += ccf[j] * tau[j];
        norm_temp += ccf[j];
      }
      tau_temp = tau_temp/(norm_temp+EPS);
    }
    else /* centroid of the major single peak */
    {
      int idx_above_max, idx_left, idx_right;
      /* starting from the index corresponding to rmax */
      idx_above_max = locate_int_low(idx_above, nabove, idx_max);

      /* go right */
      idx_right = nabove-1;
      for(i=idx_above_max+1; i<nabove; i++)
      {
        if(idx_above[i]-idx_above[i-1] > 1) 
        {
          idx_right = i-1;
          break;
        }
      }
      /* go left */
      idx_left = 0;
      for(i=idx_above_max-1; i>=0; i--)
      {
        if(idx_above[i+1]-idx_above[i] > 1) 
        {
          idx_left = i+1;
          break;
        }
      }
      tau_temp = 0.0;
      norm_temp = 0.0;
      for(i=idx_left; i<=idx_right; i++)
      {
        tau_temp  += ccf[idx_above[i]]*tau[idx_above[i]];
        norm_temp += ccf[idx_above[i]];
      }
      tau_temp = tau_temp/(norm_temp+EPS);
    }

    *tau_cent = tau_temp;
  }

  free(fintp);
  free(idx_above);
  return;
}

void ciccf_mc(double t1[], double f1[], double e1[], int n1, 
              double t2[], double f2[], double e2[], int n2,
              int ntau, double tau_beg, double tau_end, int nsim, 
              double threshold, char mode, int ignore_warning,
              double ccf_peak_mc[], double tau_peak_mc[], double tau_cent_mc[]
             )
{
  int i, j, ir, nmax, n1_sim, n2_sim;
  int *counts;
  double *t1_sim, *f1_sim, *e1_sim, *t2_sim, *f2_sim, *e2_sim;
  double *tau, *ccf;
  double rmax, tau_peak, tau_cent;
  
  /* random number generator */
  const gsl_rng_type * cpyat_gsl_T;
  gsl_rng * cpyat_gsl_r;
  cpyat_gsl_T = (gsl_rng_type *) gsl_rng_default;
  cpyat_gsl_r = gsl_rng_alloc (cpyat_gsl_T);
  
  nmax = (n1>n2?n1:n2);
  counts = (int *)malloc(nmax*sizeof(int));
  
  t1_sim = (double *)malloc(n1*sizeof(double));
  f1_sim = (double *)malloc(n1*sizeof(double));
  e1_sim = (double *)malloc(n1*sizeof(double));
  t2_sim = (double *)malloc(n2*sizeof(double));
  f2_sim = (double *)malloc(n2*sizeof(double));
  e2_sim = (double *)malloc(n2*sizeof(double));

  tau = (double *)malloc(ntau*sizeof(double));
  ccf = (double *)malloc(ntau*sizeof(double));

  for(i=0; i<nsim; i++)
  {
    if(i%(nsim/10) == 0)
      printf("%d%-", 100*i/nsim);
    
    /* resample f1 */
    for(j=0; j<n1; j++) /* re-reinitialize counts */
      counts[j] = 0;

    for(j=0; j<n1; j++)
    {
      ir = gsl_rng_uniform_int(cpyat_gsl_r, n1);
      counts[ir] += 1;
    }
    
    n1_sim = 0;
    for(j=0; j<n1; j++)
    {
      if(counts[j] > 0)
      {
        t1_sim[n1_sim] = t1[j];
        e1_sim[n1_sim] = e1[j]/sqrt(counts[j]);
        f1_sim[n1_sim] = f1[j] + e1_sim[n1_sim] * gsl_ran_ugaussian(cpyat_gsl_r);
        n1_sim++;
      }
    }

    /* resample f2 */
    for(j=0; j<n2; j++) /* re-reinitialize counts */
      counts[j] = 0;

    for(j=0; j<n2; j++)
    {
      ir = gsl_rng_uniform_int(cpyat_gsl_r, n2);
      counts[ir] += 1;
    }
    
    n2_sim = 0;
    for(j=0; j<n2; j++)
    {
      if(counts[j] > 0)
      {
        t2_sim[n2_sim] = t2[j];
        e2_sim[n2_sim] = e2[j]/sqrt(counts[j]);
        f2_sim[n2_sim] = f2[j] + e2_sim[n2_sim] * gsl_ran_ugaussian(cpyat_gsl_r);
        n2_sim++;
      }
    }

    ciccf(t1_sim, f1_sim, n1_sim, t2_sim, f2_sim, n2_sim,
          ntau, tau_beg, tau_end, threshold, mode, ignore_warning,
          tau, ccf, &rmax, &tau_peak, &tau_cent
         );
    
    ccf_peak_mc[i] = rmax;
    tau_peak_mc[i] = tau_peak;
    tau_cent_mc[i] = tau_cent;
  }
  printf("Done\n");
  
  gsl_rng_free(cpyat_gsl_r);
  free(counts);
  free(t1_sim);
  free(f1_sim);
  free(e1_sim);
  free(t2_sim);
  free(f2_sim);
  free(e2_sim);

  free(tau);
  free(ccf);
}

void ciccf_mc_oneway(double t1[], double f1[], double e1[], int n1, 
              double t2[], double f2[], double e2[], int n2,
              int ntau, double tau_beg, double tau_end, int nsim, 
              double threshold, char mode, int ignore_warning,
              double ccf_peak_mc[], double tau_peak_mc[], double tau_cent_mc[]
             )
{
  int i, j, ir, nmax, n1_sim, n2_sim;
  int *counts;
  double *t1_sim, *f1_sim, *e1_sim, *t2_sim, *f2_sim, *e2_sim;
  double *tau, *ccf;
  double rmax, tau_peak, tau_cent;
  
  /* random number generator */
  const gsl_rng_type * cpyat_gsl_T;
  gsl_rng * cpyat_gsl_r;
  cpyat_gsl_T = (gsl_rng_type *) gsl_rng_default;
  cpyat_gsl_r = gsl_rng_alloc (cpyat_gsl_T);
  
  nmax = (n1>n2?n1:n2);
  counts = (int *)malloc(nmax*sizeof(int));
  
  t1_sim = (double *)malloc(n1*sizeof(double));
  f1_sim = (double *)malloc(n1*sizeof(double));
  e1_sim = (double *)malloc(n1*sizeof(double));
  t2_sim = (double *)malloc(n2*sizeof(double));
  f2_sim = (double *)malloc(n2*sizeof(double));
  e2_sim = (double *)malloc(n2*sizeof(double));

  tau = (double *)malloc(ntau*sizeof(double));
  ccf = (double *)malloc(ntau*sizeof(double));

  for(i=0; i<nsim; i++)
  {
    if(i%(nsim/10) == 0)
      printf("%d%-", 100*i/nsim);
    
    /* resample f1 */
    for(j=0; j<n1; j++) /* re-reinitialize counts */
      counts[j] = 0;

    for(j=0; j<n1; j++)
    {
      ir = gsl_rng_uniform_int(cpyat_gsl_r, n1);
      counts[ir] += 1;
    }
    
    n1_sim = 0;
    for(j=0; j<n1; j++)
    {
      if(counts[j] > 0)
      {
        t1_sim[n1_sim] = t1[j];
        e1_sim[n1_sim] = e1[j]/sqrt(counts[j]);
        f1_sim[n1_sim] = f1[j] + e1_sim[n1_sim] * gsl_ran_ugaussian(cpyat_gsl_r);
        n1_sim++;
      }
    }

    /* resample f2 */
    for(j=0; j<n2; j++) /* re-reinitialize counts */
      counts[j] = 0;

    for(j=0; j<n2; j++)
    {
      ir = gsl_rng_uniform_int(cpyat_gsl_r, n2);
      counts[ir] += 1;
    }
    
    n2_sim = 0;
    for(j=0; j<n2; j++)
    {
      if(counts[j] > 0)
      {
        t2_sim[n2_sim] = t2[j];
        e2_sim[n2_sim] = e2[j]/sqrt(counts[j]);
        f2_sim[n2_sim] = f2[j] + e2_sim[n2_sim] * gsl_ran_ugaussian(cpyat_gsl_r);
        n2_sim++;
      }
    }

    ciccf_oneway(t1_sim, f1_sim, n1_sim, t2_sim, f2_sim, n2_sim,
          ntau, tau_beg, tau_end, threshold, mode, ignore_warning,
          tau, ccf, &rmax, &tau_peak, &tau_cent
         );
    
    ccf_peak_mc[i] = rmax;
    tau_peak_mc[i] = tau_peak;
    tau_cent_mc[i] = tau_cent;
  }
  printf("Done\n");
  
  gsl_rng_free(cpyat_gsl_r);
  free(counts);
  free(t1_sim);
  free(f1_sim);
  free(e1_sim);
  free(t2_sim);
  free(f2_sim);
  free(e2_sim);

  free(tau);
  free(ccf);
}

double mean_cal(double x[], int n)
{
  int i;
  double mean;

  mean = 0.0;
  for(i=0; i<n; i++)
  {
    mean += x[i];
  }

  return mean/n;
}

double std_cal(double x[], int n)
{
  int i; 
  double mean, std;

  mean = mean_cal(x, n);
  std = 0.0;
  for(i=0; i<n; i++)
  {
    std += (x[i]-mean)*(x[i]-mean);
  }

  return sqrt(std/n);
}

double interpolate(double t[], double x[], int n, double ti)
{
  unsigned int i, intp;
  double xi;

  intp = locate_low(t, n, ti);
  
  if(intp == n-1)
  {
    xi = x[n-1];
  }
  else 
  {
    xi = x[intp] + (x[intp+1]-x[intp])/(t[intp+1]-t[intp]) * (ti-t[intp]);
  }
  return xi;
}

/* locate the index i of xi in x,   x[i]<=xi<x[i+1] */
int locate_low(double x[], int n, double xi)
{
  int i, il, im, iu;

  if(xi == x[0])
  {
    return 0;
  }
  else if(xi == x[n-1])
  {
    return n-1;
  }
  else 
  {
    il = 0;
    iu = n;  /* this ensures that iu can take n-1 */
    while(iu - il > 1)
    {
      im = (iu+il) >> 1;
      if(xi == x[im])
        return im;
      else if(xi > x[im])
        il = im;
      else 
        iu = im;
    }
    return il;
  }
}
/* locate the index i+1 of xi in x,   x[i]<=xi<x[i+1] */
int locate_upp(double x[], int n, double xi)
{
  int i, il, im, iu;

  if(xi == x[0])
  {
    return 0;
  }
  else if(xi == x[n-1])
  {
    return n-1;
  }
  else 
  {
    il = -1;  /* this ensures that iu can take zero */
    iu = n-1;
    while(iu - il > 1)
    {
      im = (iu+il) >> 1;
      if(xi == x[im])
        return im;
      else if(xi < x[im])
        iu = im;
      else 
        il = im;
    }
    return iu;
  }
}
/* locate the index i of xi in x,   x[i]<=xi<x[i+1] */
int locate_int_low(int x[], int n, int xi)
{
  int i, il, im, iu;

  if(xi == x[0])
  {
    return 0;
  }
  else if(xi == x[n-1])
  {
    return n-1;
  }
  else 
  {
    il = 0;
    iu = n;  /* this ensures that iu can take n-1 */
    while(iu - il > 1)
    {
      im = (iu+il) >> 1;
      if(xi == x[im])
        return im;
      else if(xi > x[im])
        il = im;
      else 
        iu = im;
    }
    return il;
  }
}