#pragma once

#include <vector>
#include <list>
#include <tuple>
#include <cmath>
#include <stdexcept>
#include "global.h"
#include "v3.h"
#include "../traj/TrajFunc.h"

class GradGen
{
public:
    typedef std::vector<int64_t> vl;
    typedef std::vector<double> vd;
    typedef std::list<int64_t> ll;
    typedef std::list<double> ld;

    typedef std::vector<v3> vv3;
    typedef std::vector<vv3> vvv3;
    typedef std::list<v3> lv3;
    typedef std::list<vv3> lvv3;

    GradGen
    (
        const TrajFunc* ptTraj,
        double dSLim, double dGLim,
        double dDt=10e-6, int64_t lOs=10, 
        double dG0Norm=0e0, double dG1Norm=0e0
    );
    ~GradGen();
    bool compute(lv3* plv3G, ld* pldP=NULL);
    template <typename T>
    static bool decomp
    (
        std::vector<T>* pvfGx,
        std::vector<T>* pvfGy,
        std::vector<T>* pvfGz,
        const vv3& vv3G,
        bool bResize = false,
        bool bFillZero = true
    );
    static bool ramp_front(lv3* plv3GRamp, const v3& v3G0, const v3& v3G0Des, double dSLim, double dDt);
    static bool ramp_front(lv3* plv3GRamp, const v3& v3G0, const v3& v3G0Des, int64_t lNSamp, double dDt);
    static bool ramp_back(lv3* plv3GRamp, const v3& v3G1, const v3& v3G1Des, double dSLim, double dDt);
    static bool ramp_back(lv3* plv3GRamp, const v3& v3G1, const v3& v3G1Des, int64_t lNSamp, double dDt);
    static bool revGrad(v3* pv3M0Dst, lv3* plv3Dst, const v3& v3M0Src, const lv3& lv3Src, double dDt);
    static v3 calM0(const lv3& lv3Grad, double dDt, const v3& v3GBegin=v3(0,0,0), const v3& v3GEnd=v3(0,0,0));
private:
    const TrajFunc* m_ptTraj;
    const double m_dSLim, m_dGLim;
    const double m_dDt;
    const int64_t m_lOs;
    const double m_dG0Norm, m_dG1Norm;

    bool sovQDE(double* pdSol0, double* pdSol1, double dA, double dB, double dC);
    double getCurRad(double dP);
    double getDp(const v3& v3G, double dDt, double dP, double dSignDp);
    bool step(v3* pv3GUnit, double* pdGNormMin, double* pdGNormMax, double dP, double dSignDp, const v3& v3G, double dSLim, double dDt);
};

// definition must be in `.h` file (compiler limitation)
template <typename T>
bool GradGen::decomp
(
    std::vector<T>* pvfGx,
    std::vector<T>* pvfGy,
    std::vector<T>* pvfGz,
    const vv3& vv3G,
    bool bResize,
    bool bFillZero
)
{
    if (bResize)
    {
        pvfGx->resize(vv3G.size());
        pvfGy->resize(vv3G.size());
        pvfGz->resize(vv3G.size());
    }
    if (bFillZero)
    {
        std::fill(pvfGx->begin(), pvfGx->end(), (T)0);
        std::fill(pvfGy->begin(), pvfGy->end(), (T)0);
        std::fill(pvfGz->begin(), pvfGz->end(), (T)0);
    }
    typename std::vector<T>::iterator ivfGx = pvfGx->begin();
    typename std::vector<T>::iterator ivfGy = pvfGy->begin();
    typename std::vector<T>::iterator ivfGz = pvfGz->begin();
    vv3::const_iterator ivv3G = vv3G.begin();
    while (ivv3G != vv3G.end())
    {
        *ivfGx = T(ivv3G->m_dX);
        *ivfGy = T(ivv3G->m_dY);
        *ivfGz = T(ivv3G->m_dZ);
        ++ivfGx;
        ++ivfGy;
        ++ivfGz;
        ++ivv3G;
    }
    return true;
}