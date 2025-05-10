#include <cassert>
#include <algorithm>
#include <cstdio>
#include "GradGen.h"

// interpolation function
template<typename T>
static T intp(double dXEv, double dX0, T tY0, double dX1, T tY1)
{
    if (std::fabs(dXEv-dX0) < std::fabs(dXEv-dX1))
    {
        return tY0 + (tY1-tY0)/(dX1-dX0) * (dXEv-dX0);
    }
    else
    {
        return tY1 + (tY0-tY1)/(dX0-dX1) * (dXEv-dX1);
    }
}

GradGen::GradGen
    (
        const TrajFunc* ptTraj,
        double dSLim, double dGLim,
        double dDt, int64_t lOs, 
        double dG0Norm, double dG1Norm
    ):
    m_ptTraj(ptTraj),
    m_dSLim(dSLim), 
    m_dGLim(dGLim), 
    m_dDt(dDt), 
    m_lOs(lOs), 
    m_dG0Norm(dG0Norm), 
    m_dG1Norm(dG1Norm)
{

}

GradGen::~GradGen()
{

}

bool GradGen::sovQDE(double* pdSol0, double* pdSol1, double dA, double dB, double dC)
{
    double dDelta = dB*dB - 4*dA*dC;
    if (dDelta<0) dDelta = 0;
    if (pdSol0) *pdSol0 = (-dB-std::sqrt(dDelta))/(2*dA);
    if (pdSol1) *pdSol1 = (-dB+std::sqrt(dDelta))/(2*dA);
    return true;
}

double GradGen::getCurRad(double dP)
{
    v3 v3DkDp; m_ptTraj->getDkDp(&v3DkDp, dP);
    v3 v3D2kDp2; m_ptTraj->getD2kDp2(&v3D2kDp2, dP);
    double dNume = pow(v3::norm(v3DkDp), 3e0);
    double dDeno = v3::norm(v3::cross(v3DkDp, v3D2kDp2));
    return dNume/dDeno;
}

#if 1
double GradGen::getDp(const v3& v3G, double dDt, double dP, double dSignDp)
{
    // solve `ΔP` by RK4
    v3 v3Dk = v3G*dDt;
    double dDl = v3::norm(v3Dk)*dSignDp;
    // k1
    double dK1;
    {
        v3 v3DkDp; m_ptTraj->getDkDp(&v3DkDp, dP);
        double dDlDp = v3::norm(v3DkDp);
        dK1 = 1/dDlDp;
    }
    // k2, k3
    double dK2, dK3;
    {
        v3 v3DkDp; m_ptTraj->getDkDp(&v3DkDp, dP+dK1*dDl/2);
        double dDlDp = v3::norm(v3DkDp);
        dK2 = 1/dDlDp;
        dK3 = 1/dDlDp;
    }
    // k4
    double dK4;
    {
        v3 v3DkDp; m_ptTraj->getDkDp(&v3DkDp, dP+dK1*dDl);
        double dDlDp = v3::norm(v3DkDp);
        dK4 = 1/dDlDp;
    }
    double dDp = dDl*(dK1 + 2*dK2 + 2*dK3 + dK4)/6;
    return dDp;
}
#else
double GradGen::getDp(const v3& v3G, double dDt, double dP, double dSignDp)
{
    // solve `ΔP`
    v3 v3Dk = v3G*dDt;
    double dDl = v3::norm(v3Dk)*dSignDp;
    v3 v3DkDp; m_ptTraj->getDkDp(&v3DkDp, dP);
    double dDlDp = v3::norm(v3DkDp);
    double dDp = dDl/dDlDp;

    // correct `ΔP`
    v3 v3K0; m_ptTraj->getK(&v3K0, dP);
    v3 v3K1; m_ptTraj->getK(&v3K1, dP+dDp);
    double dDl_ = v3::norm(v3K1-v3K0)*dSignDp;
    dDp *= dDl/dDl_;

    return dDp;
}
#endif

bool GradGen::step(v3* pv3GUnit, double* pdGNormMin, double* pdGNormMax, double dP, double dSignDp, const v3& v3G, double dSLim, double dDt)
{
    // current gradient direction
    v3 v3DkDp; m_ptTraj->getDkDp(&v3DkDp, dP);
    double dDlDp = v3::norm(v3DkDp)*dSignDp;
    if (pv3GUnit) *pv3GUnit = v3DkDp/dDlDp;
    
    // current gradient magnitude
    return sovQDE
    (
        pdGNormMin, pdGNormMax,
        1e0,
        -2e0*v3::inner(v3G, *pv3GUnit),
        v3::inner(v3G, v3G) - std::pow(dSLim*dDt, 2e0)
    );
}

bool GradGen::compute(vv3* pvv3G)
{
    bool bRet = true;
    double dP0 = m_ptTraj->getP0();
    double dP1 = m_ptTraj->getP1();

    // backward
    v3 v3G1Unit; bRet &= m_ptTraj->getDkDp(&v3G1Unit, dP1);
    v3G1Unit = v3G1Unit * (dP0>dP1?1e0:-1e0);
    v3G1Unit = v3G1Unit / v3::norm(v3G1Unit);
    double dG1Norm = m_dG1Norm;
    dG1Norm = std::min(dG1Norm, m_dGLim);
    dG1Norm = std::min(dG1Norm, std::sqrt(m_dSLim*getCurRad(dP1)));
    v3 v3G1 = v3G1Unit * dG1Norm;

    ld ldP_Bac; ldP_Bac.push_back(dP1);
    lv3 lv3G_Bac; lv3G_Bac.push_back(v3G1);
    ld ldGNorm_Bac; ldGNorm_Bac.push_back(v3::norm(v3G1));
    while (1)
    {
        double dP = *ldP_Bac.rbegin();
        v3 v3G = *lv3G_Bac.rbegin();
        // update grad
        v3 v3GUnit;
        double dGNorm;
        bRet &= step(&v3GUnit, NULL, &dGNorm, dP, (dP0-dP1)/std::fabs(dP0-dP1), v3G, m_dSLim, m_dDt/m_lOs);
        dGNorm = std::min(dGNorm, m_dGLim);
        dGNorm = std::min(dGNorm, std::sqrt(m_dSLim*getCurRad(dP)));
        v3G = v3GUnit*dGNorm;

        // update para
        dP += getDp(v3G, m_dDt/m_lOs, dP, (dP0-dP1)/std::fabs(dP0-dP1));

        // stop or append
        if (std::fabs(*ldP_Bac.rbegin() - dP1) >= (1-1e-6)*std::fabs(dP0 - dP1))
        {
            break;
        }
        else
        {
            // printf("bac: dP = %lf\n", dP); // test
            if (std::isnan(dP)) throw std::runtime_error("dP = nan");
            ldP_Bac.push_back(dP);
            lv3G_Bac.push_back(v3G);
            ldGNorm_Bac.push_back(v3::norm(v3G));
        }
    }
    vd vdP_Bac(ldP_Bac.rbegin(), ldP_Bac.rend());
    vd vdGNorm_Bac(ldGNorm_Bac.rbegin(), ldGNorm_Bac.rend());

    // forward
    v3 v3G0Unit; bRet &= m_ptTraj->getDkDp(&v3G0Unit, dP0);
    v3G0Unit = v3G0Unit * (dP1>dP0?1e0:-1e0);
    v3G0Unit = v3G0Unit / v3::norm(v3G0Unit);
    double dG0Norm = m_dG0Norm;
    dG0Norm = std::min(dG0Norm, m_dGLim);
    dG0Norm = std::min(dG0Norm, std::sqrt(m_dSLim*getCurRad(dP0)));
    dG0Norm = std::min(dG0Norm, intp(dP0, vdP_Bac[0], vdGNorm_Bac[0], vdP_Bac[1], vdGNorm_Bac[1]));
    v3 v3G0 = v3G0Unit * dG0Norm;

    ld ldP; ldP.push_back(dP0);
    lv3 lv3G; lv3G.push_back(v3G0);
    ld ldGNorm_For; ldGNorm_For.push_back(v3::norm(v3G0));
    ld::reverse_iterator ildP_Bac = std::next(ldP_Bac.rbegin());
    ld::reverse_iterator ildGNorm_Bac = std::next(ldGNorm_Bac.rbegin());
    while (1)
    {
        double dP = *ldP.rbegin();
        v3 v3G = *lv3G.rbegin();

        // update grad
        v3 v3GUnit;
        double dGNorm;
        bRet &= step(&v3GUnit, NULL, &dGNorm, dP, (dP1-dP0)/std::fabs(dP1-dP0), v3G, m_dSLim, m_dDt/m_lOs);
        dGNorm = std::min(dGNorm, m_dGLim);
        dGNorm = std::min(dGNorm, std::sqrt(m_dSLim*getCurRad(dP)));

        // find index for interpolation
        while (std::fabs(dP-dP0) > std::fabs(*ildP_Bac-dP0))
        {
            if (ildP_Bac!=ldP_Bac.rend())
            {
                ++ildP_Bac;
                ++ildGNorm_Bac;
            }
            else break;
        }

        // interpolation
        dGNorm = std::min(dGNorm, intp(dP, *std::prev(ildP_Bac), *std::prev(ildGNorm_Bac), *ildP_Bac, *ildGNorm_Bac));
        v3G = v3GUnit*dGNorm;

        // update para
        dP += getDp(v3G, m_dDt/m_lOs, dP, (dP1-dP0)/std::fabs(dP1-dP0));

        // stop or append
        if (std::fabs(*ldP.rbegin() - dP0) >= (1-1e-6)*std::fabs(dP1 - dP0) || dGNorm <= 0)
        {
            break;
        }
        else
        {
            // printf("for: dP = %lf\n", dP); // test
            ldP.push_back(dP);
            lv3G.push_back(v3G);
            ldGNorm_For.push_back(v3::norm(v3G));
        }
    }
    v3G1 = (v3::norm(v3G1)!=0?v3G1/v3::norm(v3G1):v3(0,0,0)) * std::min(v3::norm(v3G1), v3::norm(*lv3G.rbegin()));

    // deoversamp the para. vec.
    {
        ld::iterator ildP = ldP.begin();
        int64_t n = ldP.size();
        for (int64_t i = 0; i < n; ++i)
        {
            if(i%m_lOs!=m_lOs/2) ildP = ldP.erase(ildP);
            else ++ildP;
        }
    }

    // derive gradient
    {
        lv3G.clear();
        ld::iterator ildP = std::next(ldP.begin());
        int64_t n = ldP.size();
        for (int64_t i = 1; i < n; ++i)
        {
            v3 v3K1; bRet &= m_ptTraj->getK(&v3K1, *ildP);
            v3 v3K0; bRet &= m_ptTraj->getK(&v3K0, *std::prev(ildP));
            lv3G.push_back((v3K1 - v3K0)/m_dDt);
            ++ildP;
        }
    }

    vv3 vv3G(lv3G.begin(), lv3G.end());
    vv3 vv3GRampFront; bRet &= GradGen::ramp_front(&vv3GRampFront, *vv3G.begin(), v3G0, m_dSLim, m_dDt);
    vv3 vv3GRampBack; bRet &= GradGen::ramp_back(&vv3GRampBack, *vv3G.rbegin(), v3G1*-1, m_dSLim, m_dDt);
    
    lvv3 lvv3FullGRO;
    lvv3FullGRO.push_back(vv3GRampFront);
    lvv3FullGRO.push_back(vv3G);
    lvv3FullGRO.push_back(vv3GRampBack);
    bRet &= GradGen::catGrad(pvv3G, lvv3FullGRO);

    return bRet;
}

bool GradGen::ramp_front(vv3* pvv3GRamp, const v3& v3G0, const v3& v3G0Des, double dSLim, double dDt)
{
    v3 v3Dg = v3G0Des - v3G0;
    int64_t lNSamp = (int64_t)std::ceil(v3::norm(v3Dg)/(dSLim*dDt));

    // derive ramp gradient
    *pvv3GRamp = vv3(lNSamp);
    for (int64_t i = 1; i < lNSamp; ++i)
    {
        pvv3GRamp->at(lNSamp-i) = v3G0 + (v3::norm(v3Dg)!=0?v3Dg/v3::norm(v3Dg):v3(0,0,0)) * (dSLim*dDt) * i;
    }
    if (lNSamp>0) pvv3GRamp->at(0) = v3G0Des;
    
    return true;
}

bool GradGen::ramp_front(vv3* pvv3GRamp, const v3& v3G0, const v3& v3G0Des, int64_t lNSamp, double dDt)
{
    v3 v3Dg = v3G0Des - v3G0;
    double dSLim = v3::norm(v3Dg)/(lNSamp*dDt);

    // derive ramp gradient
    *pvv3GRamp = vv3(lNSamp);
    for (int64_t i = 1; i < lNSamp; ++i)
    {
        pvv3GRamp->at(lNSamp-i) = v3G0 + (v3::norm(v3Dg)!=0?v3Dg/v3::norm(v3Dg):v3(0,0,0)) * (dSLim*dDt) * i;
    }
    if (lNSamp>0) pvv3GRamp->at(0) = v3G0Des;
    
    return true;
}

bool GradGen::ramp_back(vv3* pvv3GRamp, const v3& v3G1, const v3& v3G1Des, double dSLim, double dDt)
{
    v3 v3Dg = v3G1Des - v3G1;
    int64_t lNSamp = (int64_t)std::ceil(v3::norm(v3Dg)/(dSLim*dDt));

    // derive ramp gradient
    *pvv3GRamp = vv3(lNSamp);
    for (int64_t i = 1; i < lNSamp; ++i)
    {
        pvv3GRamp->at(i-1) = v3G1 + (v3::norm(v3Dg)!=0?v3Dg/v3::norm(v3Dg):v3(0,0,0)) * (dSLim*dDt) * i;
    }
    if (lNSamp>0) pvv3GRamp->at(lNSamp-1) = v3G1Des;
    
    return true;
}

bool GradGen::ramp_back(vv3* pvv3GRamp, const v3& v3G1, const v3& v3G1Des, int64_t lNSamp, double dDt)
{
    v3 v3Dg = v3G1Des - v3G1;
    double dSLim = v3::norm(v3Dg)/(lNSamp*dDt);

    // derive ramp gradient
    *pvv3GRamp = vv3(lNSamp);
    for (int64_t i = 1; i < lNSamp; ++i)
    {
        pvv3GRamp->at(i-1) = v3G1 + (v3::norm(v3Dg)!=0?v3Dg/v3::norm(v3Dg):v3(0,0,0)) * (dSLim*dDt) * i;
    }
    if (lNSamp>0) pvv3GRamp->at(lNSamp-1) = v3G1Des;
    
    return true;
}

bool GradGen::catGrad(vv3* pvv3Grad, const lvv3& lvv3GradList)
{
    int64_t lNSamp = 0;
    lvv3::const_iterator ilvv3;

    ilvv3 = lvv3GradList.begin();
    while(ilvv3 != lvv3GradList.end())
    {
        lNSamp += ilvv3->size();
        ++ilvv3;
    }

    pvv3Grad->clear();
    pvv3Grad->reserve(lNSamp);

    ilvv3 = lvv3GradList.begin();
    while(ilvv3 != lvv3GradList.end())
    {
        pvv3Grad->insert(pvv3Grad->end(), ilvv3->begin(), ilvv3->end());
        ++ilvv3;
    }

    return true;
}

bool GradGen::revGrad(v3* pv3M0Dst, vv3* pvv3Dst, const v3& v3M0Src, const vv3& vv3Src, double dDt)
{
    bool bRet = true;

    if(vv3Src.size() <= 1) bRet = false;

    // derive Total M0
    *pv3M0Dst = v3M0Src;
    for(int64_t i = 1; i < (int64_t)vv3Src.size(); ++i)
    {
        *pv3M0Dst = *pv3M0Dst + (vv3Src[i] + vv3Src[i-1])*dDt/2e0;
    }

    // reverse gradient
    *pvv3Dst = vv3(vv3Src.rbegin(), vv3Src.rend());
    for(int64_t i = 0; i < (int64_t)pvv3Dst->size(); ++i)
    {
        pvv3Dst->at(i).m_dX = -pvv3Dst->at(i).m_dX;
        pvv3Dst->at(i).m_dY = -pvv3Dst->at(i).m_dY;
        pvv3Dst->at(i).m_dZ = -pvv3Dst->at(i).m_dZ;
    }

    return bRet;
}

v3 GradGen::calM0(const vv3& vv3Grad, double dDt, const v3& v3GBegin, const v3& v3GEnd)
{
    v3 v3M0(0,0,0);
    const v3* pv3Grad = &v3GBegin;
    for(int64_t i = 0; i < (int64_t)vv3Grad.size(); ++i)
    {
        v3M0 = v3M0 + (*pv3Grad + vv3Grad[i])*dDt/2e0;
        pv3Grad = &vv3Grad[i];
    }
    v3M0 = v3M0 + (*pv3Grad + v3GEnd)*dDt/2e0;

    return v3M0;
}