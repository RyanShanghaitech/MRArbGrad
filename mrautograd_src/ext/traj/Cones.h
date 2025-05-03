#pragma once

#include "TrajFunc.h"
#include "MrTraj.h"

class Cones;

class Cones_TrajFun: public TrajFunc
{
public:
    friend Cones;

    Cones_TrajFun(double dRhoPhi, double dTht0)
    {
        m_dRhoPhi = dRhoPhi;
        m_dTht0 = dTht0;
        m_dP0 = 0e0;
        m_dP1 = 0.5e0/m_dRhoPhi;
    }

    bool getK(v3* pv3K, double dPhi) const
    {
        double dRho = m_dRhoPhi*dPhi;

        pv3K->m_dX = dRho * std::sin(m_dTht0) * std::cos(dPhi);
        pv3K->m_dY = dRho * std::sin(m_dTht0) * std::sin(dPhi);
        pv3K->m_dZ = dRho * std::cos(m_dTht0);

        return true;
    }

protected:
    double m_dRhoPhi;
    double m_dTht0;
};

class Cones: public MrTraj
{
public:
    typedef std::list<int64_t> ll;

    Cones(const GeoPara& sGeoPara, const GradPara& sGradPara, double dRhoPhi)
    {
        m_sGeoPara = sGeoPara;
        m_sGradPara = sGradPara;
        const int64_t& lNPix = m_sGeoPara.lNPix;

        m_lNSet = getNLayer_Cones(lNPix);
        m_vlNRot.resize(m_lNSet);
        m_vptfBasicTrajSet.resize(m_lNSet);
        m_vvv3BasicGradSet.resize(m_lNSet);
        m_lNAcq = 0;
        for (int i = 0; i < m_lNSet; ++i)
        {
            printf("%d/%ld\n", i, m_lNSet); // debug
            
            double dTht0 = getTht0_Cones(i, m_lNSet);
            m_vptfBasicTrajSet[i] = new Cones_TrajFun(dRhoPhi, dTht0);
            calGrad(&m_vvv3BasicGradSet[i], *m_vptfBasicTrajSet[i], m_sGradPara, 8);
            int64_t lNRot = calNRot
            (
                m_vptfBasicTrajSet[i], 
                m_vptfBasicTrajSet[i]->getP0(), 
                m_vptfBasicTrajSet[i]->getP1(),
                lNPix
            );
            m_vlNRot[i] = lNRot;
            m_lNAcq += lNRot;
        }
    }

    virtual ~Cones()
    {
        for(int64_t i = 0; i < (int64_t)m_vptfBasicTrajSet.size(); ++i)
        {
            delete m_vptfBasicTrajSet[i];
        }
    }
    
    bool getGrad(v3* pv3K0, vv3* pvv3Grad, int64_t lIAcq) const
    {
        lIAcq %= m_lNAcq;

        bool bRet = true;
        int64_t lISet=0, lIRot=0;
        int64_t _lIAcq = 0;
        for(lISet = 0; lISet < m_lNSet; ++lISet)
        {
            if(_lIAcq + m_vlNRot[lISet] >= lIAcq) break;
            else _lIAcq += m_vlNRot[lISet];
        }
        lIRot = _lIAcq - lIAcq;

        m_vptfBasicTrajSet[lISet]->getK0(pv3K0);
        *pvv3Grad = m_vvv3BasicGradSet[lISet];
        
        double dPhiInc = calRotAngInc(m_vlNRot[lISet]);
        bRet &= v3::rotate(pv3K0, 2, dPhiInc*lIRot, *pv3K0);
        bRet &= v3::rotate(pvv3Grad, 2, dPhiInc*lIRot, *pvv3Grad);

        return bRet;
    }

protected:
    double m_dRhoPhi;
    int64_t m_lNSet;
    vl m_vlNRot;
    vptf m_vptfBasicTrajSet;
    vvv3 m_vvv3BasicGradSet;

    static int64_t getNLayer_Cones(int64_t lNPix)
    {
        return (int64_t)std::ceil(lNPix*M_PI/2e0);
    }

    static double getTht0_Cones(int64_t lILayer, int64_t lNLayer)
    {
        double dDTht = M_PI / (lNLayer-1);

        return lILayer*dDTht;
    }
};
