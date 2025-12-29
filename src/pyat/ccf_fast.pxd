
cdef extern from "libccf.h":
  void ciccf(double t1[], double f1[], int n1, double t2[], double f2[], int n2,             \
              int ntau, double tau_beg, double tau_end,                                      \
              double threshold, char mode, int ignore_warning,                               \
              double tau[], double ccf[], double *rmax, double *tau_peak, double *tau_cent   \
            )

  void ciccf_oneway(double t1[], double f1[], int n1, double t2[], double f2[], int n2,      \
              int ntau, double tau_beg, double tau_end,                                      \
              double threshold, char mode, int ignore_warning,                               \
              double tau[], double ccf[], double *rmax, double *tau_peak, double *tau_cent   \
            )          
              
  void ciccf_mc(double t1[], double f1[], double e1[], int n1,                               \
                double t2[], double f2[], double e2[], int n2,                               \
                int ntau, double tau_beg, double tau_end, int nsim,                          \
                double threshold, char mode, int ignore_warning,                             \
                double ccf_peak_mc[], double tau_peak_mc[], double tau_cent_mc[]             \
               )

  void ciccf_mc_oneway(double t1[], double f1[], double e1[], int n1,                        \
                double t2[], double f2[], double e2[], int n2,                               \
                int ntau, double tau_beg, double tau_end, int nsim,                          \
                double threshold, char mode, int ignore_warning,                             \
                double ccf_peak_mc[], double tau_peak_mc[], double tau_cent_mc[]             \
               )            