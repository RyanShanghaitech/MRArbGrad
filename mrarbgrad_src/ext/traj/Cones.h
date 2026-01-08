#pragma once

#include "TrajFunc.h"
#include "MrTraj.h"

class Cones_TrajFun: public TrajFunc
{
public:
    Cones_TrajFun(f64 kRhoPhi, f64 tht0):
        TrajFunc(0,0)
    {
        m_kRhoPhi = kRhoPhi;
        m_tht0 = tht0;
        m_p0 = 0e0;
        m_p1 = 0.5e0/kRhoPhi;
    }

    bool getK(v3* k, f64 p)
    {
        if (k==NULL) return false;
        f64& kRhoPhi = m_kRhoPhi;
        f64& phi = p;
        f64& tht0 = m_tht0;
        
        f64 rho = kRhoPhi*phi;

        k->x = rho * std::sin(tht0) * std::cos(phi);
        k->y = rho * std::sin(tht0) * std::sin(phi);
        k->z = rho * std::cos(tht0);

        return true;
    }

protected:
    f64 m_kRhoPhi;
    f64 m_tht0;
};

class Cones: public MrTraj
{
public:
    Cones(const GeoPara& objGeoPara, const GradPara& objGradPara, f64 kRhoPhi):
        MrTraj(objGeoPara,objGradPara,0,0)
    {
        const i64& nPix = m_objGeoPara.nPix;

        // caluclate gradient
        m_nSet = getNLayer_Cones(nPix);
        m_vptfBaseTraj.resize(m_nSet);
        m_vvv3BaseGRO.resize(m_nSet);
        m_vv3BaseM0PE.resize(m_nSet);
        
        m_nSampMax = 0;
        for (i64 i = 0; i < m_nSet; ++i)
        {
            f64 tht0 = getTht0_Cones(i, m_nSet);
            m_vptfBaseTraj[i] = new Cones_TrajFun(kRhoPhi, tht0);
            if(!m_vptfBaseTraj[i]) throw std::runtime_error("out of memory");

            calGrad(&m_vv3BaseM0PE[i], &m_vvv3BaseGRO[i], NULL, *m_vptfBaseTraj[i], m_objGradPara);
            m_nSampMax = std::max(m_nSampMax, (i64)m_vvv3BaseGRO[i].size());
        }
        
        // list of `ISet` and `IRot`
        m_vi64NRot.resize(m_nSet);
        m_nAcq = 0;
        li64 li64SetIdx, li64RotIdx;
        for (i64 i = 0; i < m_nSet; ++i)
        {
            m_vi64NRot[i] = calNRot
            (
                m_vptfBaseTraj[i], 
                m_vptfBaseTraj[i]->getP0(), 
                m_vptfBaseTraj[i]->getP1(),
                nPix
            );

            for (i64 j = 0; j < m_vi64NRot[i]; ++j)
            {
                li64SetIdx.push_back(i);
                li64RotIdx.push_back(j);
            }

            m_nAcq += m_vi64NRot[i];
        }
        m_vi64SetIdx = vi64(li64SetIdx.begin(), li64SetIdx.end());
        m_vi64RotIdx = vi64(li64RotIdx.begin(), li64RotIdx.end());
    }

    virtual ~Cones()
    {
        for(i64 i = 0; i < (i64)m_vptfBaseTraj.size(); ++i)
        {
            delete m_vptfBaseTraj[i];
        }
    }
    
    virtual bool getGRO(vv3* pvv3GRO, i64 iAcq)
    {
        bool ret = true;
        iAcq %= m_nAcq;
        i64 iSet = m_vi64SetIdx[iAcq];
        i64 iRot = m_vi64RotIdx[iAcq];
        f64 phiStep = calRotAng(m_vi64NRot[iSet]);

        *pvv3GRO = m_vvv3BaseGRO[iSet];
        ret &= v3::rotate(pvv3GRO, 2, phiStep*iRot, *pvv3GRO);

        return ret;
    }

    virtual bool getM0PE(v3* pv3M0PE, i64 iAcq)
    {
        bool ret = true;
        iAcq %= m_nAcq;
        i64 iSet = m_vi64SetIdx[iAcq];
        i64 iRot = m_vi64RotIdx[iAcq];
        f64 phiStep = calRotAng(m_vi64NRot[iSet]);

        *pv3M0PE = m_vv3BaseM0PE[iSet];
        ret &= v3::rotate(pv3M0PE, 2, phiStep*iRot, *pv3M0PE);

        return ret;
    }

protected:
    f64 m_kRhoPhi;
    i64 m_nSet;
    vi64 m_vi64NRot;
    vi64 m_vi64SetIdx;
    vi64 m_vi64RotIdx;

    vptf m_vptfBaseTraj;
    vvv3 m_vvv3BaseGRO;
    vv3 m_vv3BaseM0PE;

    static i64 getNLayer_Cones(i64 nPix)
    {
        return (i64)std::ceil(nPix*M_PI/2e0);
    }

    static f64 getTht0_Cones(i64 iLayer, i64 nLayer)
    {
        f64 dThtInc = M_PI / (nLayer-1);
        return iLayer*dThtInc;
    }
};
