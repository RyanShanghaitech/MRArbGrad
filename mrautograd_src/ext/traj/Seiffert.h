#pragma once

#include "TrajFunc.h"
#include "MrTraj.h"
#include <vector>
#include <list>

static bool cvtXyz2Ang(double* pdTht, double* pdPhi, const v3& v3Xyz)
{
    const double& dX = v3Xyz.m_dX;
    const double& dY = v3Xyz.m_dY;
    const double& dZ = v3Xyz.m_dZ;
    double dXY = std::sqrt(dX*dX + dY*dY);
    *pdTht = std::atan2(dXY, dZ);
    *pdPhi = std::atan2(dY, dX);

    return true;
}

class Seiffert_Trajfunc: public TrajFunc
{
public:
    typedef std::vector<double> vd;
    typedef std::list<double> ld;

    Seiffert_Trajfunc(double dM, double dUMax)
    {
        m_dM = dM;
        m_dUMax = dUMax;
        m_dP0 = 0e0;
        m_dP1 = dUMax;
        m_dThtBias = 0e0; m_dPhiBias = 0e0;
        v3 v3EndPt; getK(&v3EndPt, dUMax);
        cvtXyz2Ang(&m_dThtBias, &m_dPhiBias, v3EndPt);
    }
    
    bool getK(v3* pv3K, double dU) const
    {
        double dSn, dCn;
        calJacElip(&dSn, &dCn, m_dM, dU);

        double dRho = 0.5e0 * (dU/m_dUMax);
        pv3K->m_dX = dRho * dSn * std::cos(dU*std::sqrt(m_dM));
        pv3K->m_dY = dRho * dSn * std::sin(dU*std::sqrt(m_dM));
        pv3K->m_dZ = dRho * dCn;

        v3::rotate(pv3K, 2, -m_dPhiBias, *pv3K);
        v3::rotate(pv3K, 1, -m_dThtBias, *pv3K);

        return true;
    }
    
protected:
    double m_dM, m_dUMax;
    double m_dThtBias, m_dPhiBias;
    
    static bool calJacElip(double* pdSn, double* pdCn, double dM, double dU)
    {
        if (dM<0e0 || dM>1e0)
        {
            printf("ArgError, dM=%lf\n", dM);
            abort();
        }

        ld ldA; ldA.push_back(1e0);
        ld ldB; ldB.push_back(std::sqrt(1e0-dM));
        ld ldC; ldC.push_back(0e0);
        while (std::fabs(*std::prev(ldB.end()) - *std::prev(ldA.end())) > 1e-8)
        {
            const double& dA_Old = *std::prev(ldA.end());
            const double& dB_Old = *std::prev(ldB.end());
            double dA_New = (dA_Old + dB_Old) / 2e0;
            double dB_New = std::sqrt(dA_Old * dB_Old);
            double dC_New = (dA_Old - dB_Old) / 2e0;
            ldA.push_back(dA_New);
            ldB.push_back(dB_New);
            ldC.push_back(dC_New);
        }
        int64_t lN = ldA.size() - 1;
        vd vdA(ldA.begin(), ldA.end());
        vd vdB(ldB.begin(), ldB.end());
        vd vdC(ldC.begin(), ldC.end());

        vd vdPhi(lN+1, 0e0);
        vdPhi[lN] = std::pow(2e0,double(lN)) * vdA[lN] * dU;
        for (int64_t lIdx_N = lN; lIdx_N >= 1; --lIdx_N)
        {
            vdPhi[lIdx_N-1] = (1e0/2e0)*(vdPhi[lIdx_N] + std::asin(vdC[lIdx_N]/vdA[lIdx_N]*std::sin(vdPhi[lIdx_N])));
        }

        double dAm = vdPhi[0];
        *pdSn = std::sin(dAm);
        *pdCn = std::cos(dAm);
        
        return true;
    }
};

class Seiffert: public MrTraj
{
public:
    Seiffert(const GeoPara& sGeoPara, const GradPara& sGradPara, double dM, double dUMax)
    // m = 0.07 is optimized for diaphony
    // Umax = 20 can achieve similar readout time as original paper
    {
        m_sGeoPara = sGeoPara;
        m_sGradPara = sGradPara;
        const int64_t& lNPix = m_sGeoPara.lNPix;
        m_lNAcq = (int64_t)round(-2.53819233e-03*lNPix*lNPix + 8.53447761e+01*lNPix); // fitted

        m_ptfBasicTraj = new Seiffert_Trajfunc(dM, dUMax);
        if(!m_ptfBasicTraj) throw std::runtime_error("out of memory");

        calGrad(&m_vv3BasicGrad, *m_ptfBasicTraj, m_sGradPara, 16);
    }
    
    virtual ~Seiffert()
    {
        delete m_ptfBasicTraj;
    }
    
    bool getGrad(v3* pv3K0, vv3* pvv3Grad, int64_t lIAcq) const
    {
        bool bRet = true;

        m_ptfBasicTraj->getK0(pv3K0);
        *pvv3Grad = m_vv3BasicGrad;
        
        vl vlAx; vd vdAng;
        bRet &= getRotAng(&vlAx, &vdAng, lIAcq);

        bRet &= appRotAng(pv3K0, *pv3K0, vlAx, vdAng);
        bRet &= appRotAng(pvv3Grad, *pvv3Grad, vlAx, vdAng);

        return bRet;
    }
    
protected:
    TrajFunc* m_ptfBasicTraj;
    vv3 m_vv3BasicGrad;

    bool getRotAng(vl* pvlAx, vd* pvdAng, int64_t lIAcq) const
    {
        pvlAx->resize(3);
        pvdAng->resize(3);

        // randomly rotate around z-axis
        pvlAx->at(0) = 2;
        pvdAng->at(0) = lIAcq*(lIAcq+1)*GOLDANG;

        // rotate endpoint to Fibonaci Points
        v3 v3FibPt;
        {
            int64_t lNf = m_lNAcq;
            double dK = double(lIAcq%m_lNAcq) - lNf/2;

            double dSf = dK/(lNf/2);
            double dCf = std::sqrt((lNf/2+dK)*(lNf/2-dK)) / (lNf/2);
            double dPhi = (1e0+std::sqrt(5e0)) / 2e0;
            double dTht = 2e0*M_PI*dK/dPhi;

            v3FibPt.m_dX = dCf*std::sin(dTht);
            v3FibPt.m_dY = dCf*std::cos(dTht);
            v3FibPt.m_dZ = dSf;
        }
        double dTht, dPhi; cvtXyz2Ang(&dTht, &dPhi, v3FibPt);

        pvlAx->at(1) = 1;
        pvdAng->at(1) = dTht;
        pvlAx->at(2) = 2;
        pvdAng->at(2) = dPhi;

        return true;
    }

    template<typename T>
    bool appRotAng(T* ptDst, const T& tSrc, vl vlAx, vd vdAng) const
    {
        bool bRet = true;

        if (vlAx.size() != vdAng.size()) throw std::invalid_argument("pllAx->size() != pldAng->size()");

        *ptDst = tSrc;
        for(int64_t i = 0; i < (int64_t)vlAx.size(); ++i)
        {
            bRet &= v3::rotate(ptDst, vlAx[i], vdAng[i], *ptDst);
        }

        return bRet;
    }
};
