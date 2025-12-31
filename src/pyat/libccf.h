
#ifndef _LIBCCF_H
#define _LIBCCF_H

void ciccf_proto(double t1[], double f1[], int n1, double t2[], double f2[], int n2, 
           int ntau, double tau_beg, double tau_end, 
           double threshold, char mode, int ignore_warning, int ways,
           double tau[], double ccf[], double *rmax, double *tau_peak, double *tau_cent
          );

void ciccf(double t1[], double f1[], int n1, double t2[], double f2[], int n2, 
           int ntau, double tau_beg, double tau_end, 
           double threshold, char mode, int ignore_warning,
           double tau[], double ccf[], double *rmax, double *tau_peak, double *tau_cent
          );

void ciccf_oneway(double t1[], double f1[], int n1, double t2[], double f2[], int n2, 
           int ntau, double tau_beg, double tau_end, 
           double threshold, char mode, int ignore_warning,
           double tau[], double ccf[], double *rmax, double *tau_peak, double *tau_cent
          );          

void ciccf_peak_proto(double t1[], double f1[], int n1, double t2[], double f2[], int n2,
                int ntau, double tau_beg, double tau_end, int ways,
                double tau[], double ccf[], double *rmax, int *idx_max, double *tau_peak);

void ciccf_peak(double t1[], double f1[], int n1, double t2[], double f2[], int n2,
                int ntau, double tau_beg, double tau_end, 
                double tau[], double ccf[], double *rmax, double *tau_peak);

void ciccf_oneway_peak(double t1[], double f1[], int n1, double t2[], double f2[], int n2,
                int ntau, double tau_beg, double tau_end, 
                double tau[], double ccf[], double *rmax, double *tau_peak);

void ciccf_mc_proto(double t1[], double f1[], double e1[], int n1, 
              double t2[], double f2[], double e2[], int n2,
              int ntau, double tau_beg, double tau_end, int nsim, 
              double threshold, char mode, int ignore_warning, int ways,
              double ccf_peak_mc[], double tau_peak_mc[], double tau_cent_mc[]
             );

void ciccf_mc(double t1[], double f1[], double e1[], int n1, 
              double t2[], double f2[], double e2[], int n2,
              int ntau, double tau_beg, double tau_end, int nsim, 
              double threshold, char mode, int ignore_warning,
              double ccf_peak_mc[], double tau_peak_mc[], double tau_cent_mc[]
             );

void ciccf_mc_oneway(double t1[], double f1[], double e1[], int n1, 
              double t2[], double f2[], double e2[], int n2,
              int ntau, double tau_beg, double tau_end, int nsim, 
              double threshold, char mode, int ignore_warning,
              double ccf_peak_mc[], double tau_peak_mc[], double tau_cent_mc[]
             );

int locate_low(double x[], int n, double xi);
int locate_upp(double x[], int n, double xi);
int locate_int_low(int x[], int n, int xi);
double interpolate(double t[], double x[], int n, double ti);
double mean_cal(double x[], int n);
double std_cal(double x[], int n);
#endif