#pragma once

// #ifdef __cplusplus
// extern "C" {
// #endif

void minTimeGradientRIV(const double *Ci, int Cr, int Cc, double g0, double gfin, double gmax, double smax, double T, double ds,
        double **Cx, double **Cy, double **Cz, double **gx, double **gy, double **gz, double **p_of_t,
        double **sx, double **sy, double **sz, double **kx, double **ky, double **kz, double **sdot, double **sta, double **stb, double *time,
        int *size_interpolated, int *size_sdot, int *size_st, int gfin_empty, int ds_empty);
        
int spline (int n, int end1, int end2,
            double slope1, double slope2,
            double x[], double y[],
            double b[], double c[], double d[],
            int *iflag);

double deriv2 (int n, double u,
			   double x[],
			   double b[], double c[], double d[],
			   int *last);
double sinteg (int n, double u,
			   double x[], double y[],
			   double b[], double c[], double d[],
			   int *last);
double deriv (int n, double u,
              double x[],
              double b[], double c[], double d[],
              int *last);
double seval (int n, double u,
              double x[], double y[],
              double b[], double c[], double d[],
              int *last);

// #ifdef __cplusplus
// }
// #endif