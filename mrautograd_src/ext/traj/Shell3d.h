#pragma once

#include "TrajFunc.h"
#include "MrTraj.h"

class Shell3d_TrajFunc: public TrajFunc
{
public:
    Shell3d_TrajFunc(double dRhoTht, double dTht0)
    {
        m_dThtSqrtPhi = std::sqrt(2e0);
        m_dRhoSqrtPhi = std::sqrt(2e0)*dRhoTht;
        m_dTht0 = dTht0;

        m_dP0 = 2e-4;
        m_dP1 = 1e0/(8e0*dRhoTht*dRhoTht);
    }

    ~Shell3d_TrajFunc()
    {}

    bool getK(v3* pv3K, double dP) const
    {
        const double& dPhi = dP;
        double dSqrtPhi = dPhi/std::fabs(dPhi) * std::sqrt(std::fabs(dPhi)); // odd extension
        double dRho = m_dRhoSqrtPhi * dSqrtPhi;
        double dTht = m_dThtSqrtPhi * dSqrtPhi;

        pv3K->m_dX = dRho * std::sin(dTht+m_dTht0) * std::cos(dPhi);
        pv3K->m_dY = dRho * std::sin(dTht+m_dTht0) * std::sin(dPhi);
        pv3K->m_dZ = dRho * std::cos(dTht+m_dTht0);

        return true;
    }
protected:
    double m_dThtSqrtPhi, m_dRhoSqrtPhi;
    double m_dTht0;
};

class Shell3d: public MrTraj
{
public:
    Shell3d(const GeoPara& sGeoPara, const GradPara& sGradPara, double dRhoTht)
    {
        m_sGeoPara = sGeoPara;
        m_sGradPara = sGradPara;
        m_lNRot = calNRot(dRhoTht, m_sGeoPara.lNPix);
        m_dRotInc = calRotAngInc(m_lNRot);
        m_lNAcq = m_lNRot*m_lNRot;
        
        m_vptfBasicTrajSet.resize(m_lNRot);
        m_vvv3BasicGradSet.resize(m_lNRot);
        for(int64_t i = 0; i < m_lNRot; ++i)
        {
            printf("%ld/%ld\n", i, m_lNRot); // debug

            double dTht0 = i*m_dRotInc;
            m_vptfBasicTrajSet[i] = new Shell3d_TrajFunc(dRhoTht, dTht0);
            if(!m_vptfBasicTrajSet[i]) throw std::runtime_error("out of memory");

            calGrad(&m_vvv3BasicGradSet[i], *m_vptfBasicTrajSet[i], m_sGradPara, 8);
        }
    }
    
    virtual ~Shell3d()
    {
        for(int64_t i = 0; i < (int64_t)m_vptfBasicTrajSet.size(); ++i)
        {
            delete m_vptfBasicTrajSet[i];
        }
    }

    bool getGrad(v3* pv3K0, vv3* pvv3Grad, int64_t lIAcq) const
    {
        bool bRet = true;
        const double& dPhiInc = m_dRotInc;
        int64_t lISet = lIAcq%m_lNRot;
        int64_t lIRot = lIAcq/m_lNRot;

        m_vptfBasicTrajSet[lISet]->getK0(pv3K0);
        // *pv3K0 = v3(0,0,0);
        *pvv3Grad = m_vvv3BasicGradSet[lISet];
        bRet &= v3::rotate(pv3K0, 2, dPhiInc*lIRot, *pv3K0);
        bRet &= v3::rotate(pvv3Grad, 2, dPhiInc*lIRot, *pvv3Grad);
        
        return bRet;
    }
    
protected:
    int64_t m_lNRot;
    double m_dRotInc;
    vptf m_vptfBasicTrajSet;
    vvv3 m_vvv3BasicGradSet;
};
