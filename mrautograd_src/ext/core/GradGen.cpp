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
        double dG0Norm, double dG1Norm,
        bool bWithEndPt
    ):
    m_ptTraj(ptTraj),
    m_dSLim(dSLim), 
    m_dGLim(dGLim), 
    m_dDt(dDt), 
    m_lOs(lOs), 
    m_dG0Norm(dG0Norm), 
    m_dG1Norm(dG1Norm), 
    m_bWithEndPt(bWithEndPt)
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

#if 0
double GradGen::getDp(const v3& v3G, double dDt, double dP, double dSignDp)
{
    // solve `ΔP` by RK4
    v3 v3Dk = v3G*dDt;
    double dDl = v3::norm(v3Dk)*dSignDp;
    // k1
    double dK1;
    {
        v3 v3DkDp; getDkDp_Num(&v3DkDp, dP);
        double dDlDp = v3::norm(v3DkDp);
        dK1 = 1/dDlDp;
    }
    // k2, k3
    double dK2, dK3;
    {
        v3 v3DkDp; getDkDp_Num(&v3DkDp, dP+dK1*dDl/2);
        double dDlDp = v3::norm(v3DkDp);
        dK2 = 1/dDlDp;
        dK3 = 1/dDlDp;
    }
    // k4
    double dK4;
    {
        v3 v3DkDp; getDkDp_Num(&v3DkDp, dP+dK1*dDl);
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
    double dP0 = m_ptTraj->getP0();
    double dP1 = m_ptTraj->getP1();

    // backward
    v3 v3G1Unit; m_ptTraj->getDkDp(&v3G1Unit, dP1);
    v3G1Unit = v3G1Unit * (dP0>dP1?1e0:-1e0);
    v3G1Unit = v3G1Unit / v3::norm(v3G1Unit);
    v3 v3G1 = v3G1Unit * std::min(m_dG1Norm, std::sqrt(m_dSLim*getCurRad(dP1)));

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
        step(&v3GUnit, NULL, &dGNorm, dP, (dP0-dP1)/std::fabs(dP0-dP1), v3G, m_dSLim, m_dDt/m_lOs);
        dGNorm = std::min(dGNorm, m_dGLim);
        dGNorm = std::min(dGNorm, std::sqrt(m_dSLim*getCurRad(dP)));
        v3G = v3GUnit*dGNorm;

        // update para
        dP += getDp(v3G, m_dDt/m_lOs, dP, (dP0-dP1)/std::fabs(dP0-dP1));

        // stop or append
        if (std::fabs(*ldP_Bac.rbegin() - dP1) >= 0.999*std::fabs(dP0 - dP1))
        {
            break;
        }
        else
        {
            // printf("dP = %lf\n", dP);
            ldP_Bac.push_back(dP);
            lv3G_Bac.push_back(v3G);
            ldGNorm_Bac.push_back(v3::norm(v3G));
        }
    }
    vd vdP_Bac(ldP_Bac.rbegin(), ldP_Bac.rend());
    vd vdGNorm_Bac(ldGNorm_Bac.rbegin(), ldGNorm_Bac.rend());

    // forward
    v3 v3G0Unit; m_ptTraj->getDkDp(&v3G0Unit, dP0);
    v3G0Unit = v3G0Unit * (dP1>dP0?1e0:-1e0);
    v3G0Unit = v3G0Unit / v3::norm(v3G0Unit);
    v3 v3G0 = v3G0Unit * std::min(m_dG0Norm, std::sqrt(m_dSLim*getCurRad(dP0)));

    ld ldP; ldP.push_back(dP0);
    lv3 lv3G; lv3G.push_back(v3G0);
    int iIPv = 0, iINx = 1;
    while (1)
    {
        double dP = *ldP.rbegin();
        v3 v3G = *lv3G.rbegin();

        // update grad
        v3 v3GUnit;
        double dGNorm;
        step(&v3GUnit, NULL, &dGNorm, dP, (dP1-dP0)/std::fabs(dP1-dP0), v3G, m_dSLim, m_dDt/m_lOs);
        dGNorm = std::min(dGNorm, m_dGLim);
        dGNorm = std::min(dGNorm, std::sqrt(m_dSLim*getCurRad(dP)));

        // find index for interpolation
        while (std::fabs(dP-dP0) > std::fabs(vdP_Bac[iINx]-dP0))
        {
            if (iINx < (int)vdP_Bac.size()-1) ++iINx;
            else break;
        }
        iIPv = iINx-1;

        // interpolation
        dGNorm = std::min(dGNorm, intp(dP, vdP_Bac[iIPv], vdGNorm_Bac[iIPv], vdP_Bac[iINx], vdGNorm_Bac[iINx]));
        v3G = v3GUnit*dGNorm;

        // update para
        dP += getDp(v3G, m_dDt/m_lOs, dP, (dP1-dP0)/std::fabs(dP1-dP0));

        // stop or append
        if (std::fabs(*ldP.rbegin() - dP0) >= 0.999*std::fabs(dP1 - dP0) || dGNorm <= 0)
        {
            break;
        }
        else
        {
            // printf("dP = %lf\n", dP);
            ldP.push_back(dP);
            lv3G.push_back(v3G);
        }
    }

    // deoversamp the para. vec.
    {
        ld::iterator ildP = ldP.begin();
        int64_t n = ldP.size();
        for (int64_t i = 0; i < n; ++i)
        {
            if(i%m_lOs!=0) ildP = ldP.erase(ildP);
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
            v3 v3K1; m_ptTraj->getK(&v3K1, *ildP);
            v3 v3K0; m_ptTraj->getK(&v3K0, *std::prev(ildP));
            lv3G.push_back((v3K1 - v3K0)/m_dDt);
            ++ildP;
        }
    }
    *pvv3G = vv3(lv3G.begin(), lv3G.end());

    // ensures exact G0 and G1
    if (m_bWithEndPt)
    {
        ramp_front(pvv3G, v3G0, m_dSLim, m_dDt);
        ramp_back(pvv3G, v3G1, m_dSLim, m_dDt);
    }

    return true;
}

bool GradGen::ramp_front(vv3* pvv3G, const v3& v3G0, double dSLim, double dDt)
{
    vv3 vv3G = *pvv3G;
    v3 v3Dg = v3G0 - *vv3G.begin();
    int64_t lNSamp = (int64_t)std::ceil(v3::norm(v3Dg)/(dSLim*dDt));

    // derive ramp gradient
    vv3 vv3Ramp(lNSamp);
    for (int64_t i = 0; i < lNSamp; ++i)
    {
        vv3Ramp[lNSamp-1-i] = *vv3G.begin() + (v3Dg/v3::norm(v3Dg)) * (dSLim*dDt) * (i+1);
    }
    vv3Ramp[0] = v3G0;

    // concat gradient
    *pvv3G = vv3(0);
    pvv3G->reserve(lNSamp+vv3G.size());
    pvv3G->insert(pvv3G->end(), vv3Ramp.begin(), vv3Ramp.end());
    pvv3G->insert(pvv3G->end(), vv3G.begin(), vv3G.end());
    return true;
}

bool GradGen::ramp_back(vv3* pvv3G, const v3& v3G1, double dSLim, double dDt)
{
    vv3 vv3G = *pvv3G;
    v3 v3Dg = v3G1 - *vv3G.rbegin();
    int64_t lNSamp = (int64_t)std::ceil(v3::norm(v3Dg)/(dSLim*dDt));

    // derive ramp gradient
    vv3 vv3Ramp(lNSamp);
    for (int64_t i = 0; i < lNSamp; ++i)
    {
        vv3Ramp[i] = *vv3G.rbegin() + (v3Dg/v3::norm(v3Dg)) * (dSLim*dDt) * (i+1);
    }
    vv3Ramp[lNSamp-1] = v3G1;

    // concat gradient
    *pvv3G = vv3(0);
    pvv3G->reserve(vv3G.size()+lNSamp);
    pvv3G->insert(pvv3G->end(), vv3G.begin(), vv3G.end());
    pvv3G->insert(pvv3G->end(), vv3Ramp.begin(), vv3Ramp.end());
    return true;
}
