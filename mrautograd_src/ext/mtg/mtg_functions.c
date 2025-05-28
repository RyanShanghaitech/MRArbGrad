/*This file contains all the functions used to calculated the time optimal gradient waveforms.

minTimeGradientRIV   -   Computes the rotationally invariant solution
     RungeKutte_riv  -   Used to solve the ODE using RK4
     beta            -   calculates sqrt (gamma^2 * smax^2 - k^2 * st^4) in the ODE. Used in RungeKutte_riv
minTimeGradientRV    -   Computes the rotationally variant solution
     RungeKutte_rv   -   Used to solve the ODE using RK4 */

#include <float.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "header.h"
#include <time.h>
#include <sys/types.h>

double beta(double k, double st, double smax) {
    /* calculates sqrt (gamma^2 * smax^2 - k^2 * st^4) used in RK4 method for rotationally invariant ODE solver */
    double gamma = 4.257;
    return  sqrt(sqrt((gamma*gamma*smax*smax - k*k*st*st*st*st)*(gamma*gamma*smax*smax - k*k*st*st*st*st)));
}

double RungeKutte_riv(double ds, double st, double k[], double smax) {
    /*  Solves ODE for rotationally invariant solution using Runge-Kutte method*/
    double k1 = ds * (1/st) * beta(k[0], st, smax);
    double k2 = ds * 1 / (st + k1/2) * beta(k[1], st + k1/2, smax);
    double k3 = ds * 1 / (st + k2/2) * beta(k[1], st + k2/2, smax);
    double k4 = ds * 1 / (st + k3/2) * beta(k[2], st + k3/2, smax);
    double rtn = k1/6 + k2/3 + k3/3 + k4/6;
    return rtn;
}

void minTimeGradientRIV(double *Ci, int Cr, int Cc, double g0, double gfin, double gmax, double smax, double T, double ds,
        double **Cx, double **Cy, double **Cz, double **gx, double **gy, double **gz, double **p_of_t,
        double **sx, double **sy, double **sz, double **kx, double **ky, double **kz, double **sdot, double **sta, double **stb, double *time,
        int *size_interpolated, int *size_sdot, int *size_st, int gfin_empty, int ds_empty) {
    
    /*Finds the time optimal gradient waveforms for the rotationally invariant constraints case.
    
    Ci           -       The input curve (Nx3 double array)
    Cr           -       row dimension of Ci
    Cc           -       column dimension of Ci
    g0           -       Initial gradient amplitude.
    gfin         -       Gradient value at the end of the trajectory.
                         If given value is not possible
                         the result would be the largest possible amplitude.
    gmax         -       Maximum gradient [G/cm] (4 default)
    smax         -       Maximum slew [G/cm/ms] (15 default)
    T            -       Sampling time intervale [ms] (4e-3 default)
    gx, gy, gz   -       pointers to gradient waveforms to be returned
    kx, ky, kz   -       pointers to k-space trajectory after interpolation
    sx, sy, sz   -       pointers to slew waveforms to be returned
    sdot         -       geometry constrains on the amplitude vs. arclength
    sta          -       pointer to solution for the forward ODE to be returned
    stb          -       pointer to solution for the backward ODE to be returned
    size_interpolated -  Dimension of interpolated k-space trajectory (kx, ky, kz) needed for creating mex return arrays.
    size_sdot    -       Dimension of sdot needed for creating mex return arrays.
    size_st      -       Dimension of sta and stb, need for creating mex return array.
    gfin_empty   -       Indicats wheter or not the final gradient amplitude was specifed */
    
    int i = 0;
    /* iflag used in spline method to signal error */
    int *iflag;
    int iflagp;
    iflag = &iflagp;
    
    double dt = T;
    double gamma = 4.257;
    
    /* Length of the curve in p-parameterization */
    int Lp = Cr;

    double *x, *y, *z;
    /* Ci given as Nx3 array, parse into x, y, z components */
    x = malloc(Cr * sizeof(double));
    y = malloc(Cr * sizeof(double));
    z = malloc(Cr * sizeof(double));
    
    for(i=0; i < Cr; i++) {
        x[i] = Ci[i];
        y[i] = Ci[i+Cr];
        if(Cc == 2) {
            z[i] = 0;       /* if inputed curve is Nx2, z = 0 */
        } else {
            z[i] = Ci[i+2*Cr];
        }
    }
    
    double p [Lp];
    
    /* Representing the curve with parameter p */
    
    for (i = 0; i < Lp; i++) {
        p[i] = i;
    }
    
    /* Interpolation of curve for gradient accuracy, using cubic spline interpolation */
    double *c1x, *c2x, *c3x,
            *c1y, *c2y, *c3y,
            *c1z, *c2z, *c3z;       /* arrays used by spline function to store coefficients. */
    
    c1z = malloc(Lp * sizeof(double));
    c2z = malloc(Lp * sizeof(double));
    c3z = malloc(Lp * sizeof(double));
    
    c1x = malloc(Lp * sizeof(double));
    c2x = malloc(Lp * sizeof(double));
    c3x = malloc(Lp * sizeof(double));
    
    c1y = malloc(Lp * sizeof(double));
    c2y = malloc(Lp * sizeof(double));
    c3y = malloc(Lp * sizeof(double));
    
    spline(Lp, 0, 0, 1, 1, p, x, c1x, c2x, c3x, iflag);
    spline(Lp, 0, 0, 1, 1, p, y, c1y, c2y, c3y, iflag);
    spline(Lp, 0, 0, 1, 1, p, z, c1z, c2z, c3z, iflag);
    
    double dp = 0.1;
    int num_evals = (int) floor((Lp-1) / dp)+ 1;
    
    double *CCx, *CCy, *CCz;
    CCx = malloc(num_evals * sizeof(double));
    CCy = malloc(num_evals * sizeof(double));
    CCz = malloc(num_evals * sizeof(double));
    
    double toeval = 0;
    
    int *last;
    int holder = 0;
    last = &holder;
    
    double *Cpx, *Cpy, *Cp_abs, *Cpz;               /* interpolated curve in p-parameterization */
    Cpx = malloc(num_evals * sizeof(double));
    Cpy =  malloc(num_evals * sizeof(double));
    Cpz = malloc(num_evals * sizeof(double));
    Cp_abs =  malloc(num_evals * sizeof(double));
    
    for (i = 0; i < num_evals; i++) {
        toeval = (double) i * dp;
        CCx[i] = seval(Lp, toeval, p, x, c1x, c2x, c3x, last);
        CCy[i] = seval(Lp, toeval, p, y, c1y, c2y, c3y, last);
        CCz[i] = seval(Lp, toeval, p, z, c1z, c2z, c3z, last);
        Cpx[i] = deriv(Lp, toeval, p, c1x, c2x, c3x, last);
        Cpy[i] = deriv(Lp, toeval, p, c1y, c2y, c3y, last);
        Cpz[i] = deriv(Lp, toeval, p, c1z, c2z, c3z, last);
        Cp_abs[i] = sqrt(Cpx[i]*Cpx[i] + Cpy[i]*Cpy[i] + Cpz[i]*Cpz[i]);
    }
    free(Cpx);    free(Cpy);    free(Cpz);
    
    /* converting to arc-length parameterization from p, using trapezoidal integration */
    
    double *s_of_p;
    s_of_p = malloc(num_evals * sizeof(double));
    s_of_p[0] = 0;
    
    double sofar = 0;
    
    for (i=0; i < num_evals; i++) {
        
        sofar += (Cp_abs[i]+ Cp_abs[i-1])/2;
        s_of_p[i] =  dp * sofar;
    }
    
    free(Cp_abs);
    
    /* length of the curve */
    double L = s_of_p[num_evals-1];
    
    /* decide ds and compute st for the first point */
    double stt0 = gamma*smax;   /* always assumes first point is max slew */
    double st0 = (stt0*dt)/2;   /* start at half the gradient for accuracy close to g=0 */
    double s0 = st0*dt;

    if (ds_empty == 1) {        /* if a ds value was not specified */
        ds = s0/1.5;     /* smaller step size for numerical accuracy */
    }
    
    int length_of_s =  (int) floor(L/ds);
    int half_ls = (int) floor(L/(ds/2));

    *size_sdot = half_ls;
    
    double *s;
    s = malloc(length_of_s * sizeof(double));
    sta[0] =  malloc(length_of_s * sizeof(double));
    stb[0] =  malloc(length_of_s * sizeof(double));
    
    *size_st= length_of_s;
    
    for (i = 0; i< length_of_s; i++) {
        s[i] = i*ds;
        sta[0][i] = 0;
        stb[0][i] = 0;
    }
    double *s_half =  malloc(half_ls*sizeof(double));
    
    for (i=0; i < half_ls; i++) {
        s_half[i] = (double)i*(ds/2);
    }
    
    double *p_of_s_half;
    p_of_s_half = malloc(half_ls * sizeof(double));
    
    /* Convert from s(p) to p(s) and interpolate for accuracy */
    double *a1x, *a2x, *a3x;
    a1x = malloc(num_evals * sizeof(double));
    a2x = malloc(num_evals * sizeof(double));
    a3x = malloc(num_evals * sizeof(double));
    
    double sop_num[num_evals];
    for (i=0; i<num_evals; i++) {
        sop_num[i] = i * dp;
    }
    
    spline(num_evals, 0, 0, 1, 1, s_of_p, sop_num, a1x, a2x, a3x, iflag);
    
    for (i=0; i < half_ls; i++) {
        p_of_s_half[i] = seval(num_evals, s_half[i], s_of_p, sop_num, a1x, a2x, a3x, last);
    }
    
    free(a1x); free(a2x); free(a3x);
    free(s_of_p);
    
    int size_p_of_s = half_ls/2;
    
    double *p_of_s;
    p_of_s = malloc(size_p_of_s * sizeof(double));
    
    for (i=0; i<size_p_of_s; i++) {
        p_of_s[i] = p_of_s_half[2*i];
    }
    
    double *k;
    k = malloc(half_ls*sizeof(double));        /* k is the curvature along the curve */
    
    double *Cspx, *Cspy, *Cspz;
    /* Csp is C(s(p)) = [Cx(p(s)) Cy(p(s)) Cz(p(s))] */
    Cspx =  malloc(length_of_s*sizeof(double));
    Cspy =  malloc(length_of_s*sizeof(double));
    Cspz =  malloc(length_of_s*sizeof(double));
    
    for (i=0; i<length_of_s; i++) {
        Cspx[i] = seval(Lp, p_of_s[i], p, x, c1x, c2x, c3x, last);
        Cspy[i] = seval(Lp, p_of_s[i], p, y, c1y, c2y, c3y, last);
        Cspz[i] = seval(Lp, p_of_s[i], p, z, c1z, c2z, c3z, last);
    }
    
    double *Csp1x, *Csp2x, *Csp3x, *Csp1y, *Csp2y, *Csp3y,  *Csp1z, *Csp2z, *Csp3z; /* arrays used by spline function to store coefficients. */
    Csp1x =  malloc(length_of_s*sizeof(double));
    Csp2x =  malloc(length_of_s*sizeof(double));
    Csp3x =  malloc(length_of_s*sizeof(double));
    Csp1y =  malloc(length_of_s*sizeof(double));
    Csp2y =  malloc(length_of_s*sizeof(double));
    Csp3y =  malloc(length_of_s*sizeof(double));
    Csp1z =  malloc(length_of_s*sizeof(double));
    Csp2z =  malloc(length_of_s*sizeof(double));
    Csp3z =  malloc(length_of_s*sizeof(double));
    spline(length_of_s, 0, 0, 1, 1, s, Cspx, Csp1x, Csp2x, Csp3x, iflag);
    spline(length_of_s, 0, 0, 1, 1, s, Cspy, Csp1y, Csp2y, Csp3y, iflag);
    spline(length_of_s, 0, 0, 1, 1, s, Cspz, Csp1z, Csp2z, Csp3z, iflag);
    
    for (i=0; i<half_ls; i++) {
        double kx = (deriv2(length_of_s, s_half[i], s, Csp1x, Csp2x, Csp3x, last));
        double ky = (deriv2(length_of_s, s_half[i], s, Csp1y, Csp2y, Csp3y, last));
        double kz = (deriv2(length_of_s, s_half[i], s, Csp1z, Csp2z, Csp3z, last));
        k[i] =  sqrt(kx*kx + ky*ky + kz*kz);        /* the curvature, magnitude of the second derivative of the curve in arc-length parameterization */
    }
    
    free(CCx);    free(CCy);    free(CCz);
    free(p_of_s_half);
    
    /* computing geomtry dependent constraints (forbidden line curve) */
    
    double *sdot1, *sdot2;
    sdot1 =  malloc(half_ls*sizeof(double));
    sdot2 =  malloc(half_ls*sizeof(double));
    
    sdot[0] =  malloc(half_ls * sizeof(double));
    *size_sdot = half_ls;
    /* Calculating the upper bound for the time parametrization */
    /* sdot (which is a non scaled max gradient constaint) as a function of s. */
    /* sdot is the minimum of gamma*gmax and sqrt(gamma*gmax / k) */
    
    for (i=0; i< half_ls; i++) {
        sdot1[i] = gamma*gmax;
        sdot2[i] = sqrt((gamma*smax) / (fabs(k[i]+(DBL_EPSILON))));
        if (sdot1[i] < sdot2[i]) {
            sdot[0][i] = sdot1[i];
        }else {
            sdot[0][i] = sdot2[i];
        }
    }
    free(sdot1);
    free(sdot2);
    free(s_half);
    free(Cspx); free(Cspy); free(Cspz);
    free(Csp1x); free(Csp2x); free(Csp3x);
    free(Csp1y); free(Csp2y); free(Csp3y);
    free(Csp1z); free(Csp2z); free(Csp3z);
    
    int size_k2 = half_ls+2;    /* extend of k for RK4 */
    double *k2;
    k2 = malloc(size_k2*sizeof(double));
    
    for(i=0; i < half_ls; i++) {
        k2[i] = k[i];
    }
    
    k2[size_k2-2] = k2[size_k2-3];
    k2[size_k2-1] = k2[size_k2-3];
    
    double g0gamma = g0*gamma + st0;
    double gammagmax = gamma *gmax;
    
    if (g0gamma < gammagmax) {
        sta[0][0] = g0gamma;
    } else {
        sta[0][0] = gammagmax;
    }
    
    /* Solving ODE Forward */
    for (i=1; i<length_of_s; i++) {
        double k_rk[3];
        k_rk[0] = k2[2*i-2];
        k_rk[1] = k2[2*i-1];
        k_rk[2] = k2[2*i];
        
        double dstds = RungeKutte_riv(ds, sta[0][i-1], k_rk, smax);
        double tmpst = sta[0][i-1] + dstds;
        if (sdot[0][2*i+1] < tmpst) {
            sta[0][i] = sdot[0][2*i+1];
        } else {
            sta[0][i] = tmpst;
        }
    }
    
    free(k2);
    
    /*Solving ODE Backwards: */
    double max;
    if(gfin_empty == 1) {
        /*if gfin is not provided */
        stb[0][length_of_s-1] = sta[0][length_of_s - 1];
    } else {
        
        if (gfin * gamma > st0) {
            max = gfin*gamma;
        } else {
            max = st0;
        }
        
        if (gamma * gmax < max) {
            stb[0][length_of_s-1] = gamma * gmax;
        } else {
            stb[0][length_of_s-1] = max;
        }
    }
    
    for (i=length_of_s-2; i>-1; i--) {
        double k_rk[3];
        k_rk[0] = k[2*i+2];
        k_rk[1] = k[2*i+1];
        k_rk[2] = k[2*i];
        
        double dstds = RungeKutte_riv(ds, stb[0][i+1], k_rk, smax);
        double tmpst = stb[0][i+1] + dstds;
        
        if (sdot[0][2*i] < tmpst) {
            stb[0][i] = sdot[0][2*i];
        } else {
            stb[0][i] = tmpst;
        }
    }
    
    /* take st(s) to be the minimum of the curves sta and stb */
    double *st_of_s, *st_ds_i;
    st_of_s = malloc(length_of_s*sizeof(double));
    st_ds_i = malloc(length_of_s*sizeof(double));
    
    for (i=0; i<length_of_s; i++) {
        if (sta[0][i] < stb[0][i]) {
            st_of_s[i] = sta[0][i];
        }
        else {
            st_of_s[i] = stb[0][i];
        }
        
        st_ds_i[i] = ds*(1/st_of_s[i]);         /* ds * 1/st(s) used in below calculation of t(s) */
    }
    
    /*Final interpolation */
    
    /* Converting to the time parameterization, t(s) using trapezoidal integration. t(s) = integral (1/st) ds */
    double *t_of_s= malloc(length_of_s*sizeof(double));
    t_of_s[0] = 0;
    for (i=1; i < length_of_s; i++) {
        t_of_s[i] =  t_of_s[i-1] + (st_ds_i[i]+ st_ds_i[i-1])/2;
    }
    
    int l_t =  (int) floor(t_of_s[length_of_s-1]/dt);
    *size_interpolated = l_t;       /* size of the interpolated trajectory */
    
    double t[l_t];
    for (i=0; i<l_t; i++) {
        t[i] = i*dt;                /* time array */
    }
    
    double *t1x, *t2x, *t3x;        /* coefficient arrays for spline interpolation of t(s) to get s(t) */
    
    t1x = malloc(length_of_s*sizeof(double));
    t2x = malloc(length_of_s*sizeof(double));
    t3x = malloc(length_of_s*sizeof(double));
    
    double *s_of_t;
    s_of_t = malloc(l_t * sizeof(double));
    
    spline(length_of_s, 0, 0, 1, 1, t_of_s, s, t1x, t2x, t3x, iflag);
    
    for (i=0; i < l_t; i++){
        s_of_t[i] = seval(length_of_s, t[i], t_of_s, s, t1x, t2x, t3x, last);
    }
    
    free(t1x);    free(t2x);    free(t3x);
    free(st_ds_i);
    free(st_of_s);
    free(t_of_s);
    
    double *p1x, *p2x, *p3x;        /* coefficient arrays for spline interpolation of p(s) with s(t) to get p(s(t)) = p(t) */
    p1x = malloc(length_of_s*sizeof(double));
    p2x = malloc(length_of_s*sizeof(double));
    p3x = malloc(length_of_s*sizeof(double));
    
    spline(length_of_s, 0, 0, 1, 1, s, p_of_s, p1x, p2x, p3x, iflag);
    
    p_of_t[0] = malloc(l_t*sizeof(double));
    
    for (i=0; i < l_t; i++){
        p_of_t[0][i] = seval(length_of_s, s_of_t[i], s, p_of_s, p1x, p2x, p3x, last);
    }
    
    free(s);
    free(p_of_s);
    free(p1x);    free(p2x);    free(p3x);
    free(s_of_t);
    
    /*  interpolated k-space trajectory */
    
    Cx[0] =  malloc(l_t * sizeof(double));
    Cy[0] =  malloc(l_t * sizeof(double));
    Cz[0] =  malloc(l_t * sizeof(double));
    
    for (i=0; i<l_t; i++) {
        Cx[0][i] = seval(Lp, p_of_t[0][i], p, x, c1x, c2x, c3x, last);
        Cy[0][i] = seval(Lp, p_of_t[0][i], p, y, c1y, c2y, c3y, last);
        Cz[0][i] = seval(Lp, p_of_t[0][i], p, z, c1z, c2z, c3z, last);
    }
    
    free(x);  free(y);  free(z);
    free(c1x);    free(c2x);    free(c3x);
    free(c1y);    free(c2y);    free(c3y);
    free(c1z);    free(c2z);    free(c3z);
    
    /* Final gradient waveforms to be returned */
    gx[0] =  malloc(l_t * sizeof(double));
    gy[0] =  malloc(l_t * sizeof(double));
    gz[0] =  malloc(l_t * sizeof(double));
    
    for (i=0; i< l_t -1; i++) {
        gx[0][i] = (Cx[0][i+1] - Cx[0][i]) / (gamma * dt);
        gy[0][i] = (Cy[0][i+1] - Cy[0][i]) / (gamma * dt);
        gz[0][i] = (Cz[0][i+1] - Cz[0][i]) / (gamma * dt);
    }
    
    gx[0][l_t-1] = gx[0][l_t-2] + gx[0][l_t-2] - gx[0][l_t-3];
    gy[0][l_t-1] = gy[0][l_t-2] + gy[0][l_t-2] - gy[0][l_t-3];
    gz[0][l_t-1] = gz[0][l_t-2] + gz[0][l_t-2] - gz[0][l_t-3];
    
    
    /* k-space trajecoty to be returned (calculated by integrating gradient waveforms by trapezoidal integration) */
    kx[0] =  malloc(l_t * sizeof(double));
    ky[0] =  malloc(l_t * sizeof(double));
    kz[0] =  malloc(l_t * sizeof(double));
    
    double sofarx = 0;
    double sofary = 0;
    double sofarz = 0;
    
    kx[0][0] = 0;
    ky[0][0] = 0;
    kz[0][0] = 0;
    
    for (i=1; i < l_t; i++) {
        sofarx += (gx[0][i] + gx[0][i-1]) / 2;
        sofary += (gy[0][i] + gy[0][i-1]) / 2;
        sofarz += (gz[0][i] + gz[0][i-1]) / 2;
        kx[0][i] = sofarx * dt * gamma;
        ky[0][i] = sofary * dt * gamma;
        kz[0][i] = sofarz * dt * gamma;
    }
    free(k);
    /* slew waveforms to be returned */
    sx[0] =  malloc(l_t * sizeof(double));
    sy[0] =  malloc(l_t * sizeof(double));
    sz[0] =  malloc(l_t * sizeof(double));
    
    for (i=0; i < l_t-1; i++) {
        sx[0][i] = (gx[0][i+1] - gx[0][i])/dt;
        sy[0][i] = (gy[0][i+1] - gy[0][i])/dt;
        sz[0][i] = (gz[0][i+1] - gz[0][i])/dt;
    }
    sx[0][l_t-1] = sx[0][l_t-2];
    sy[0][l_t-1] = sy[0][l_t-2];
    sz[0][l_t-1] = sz[0][l_t-2];
    
    /* total traversal time */
    *time = t[l_t-1];
}
