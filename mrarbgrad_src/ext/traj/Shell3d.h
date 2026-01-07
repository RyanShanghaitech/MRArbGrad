#pragma once

#include "TrajFunc.h"
#include "MrTraj.h"

class Shell3d_TrajFunc: public TrajFunc
{
public:
    Shell3d_TrajFunc(f64 kRhoTht, f64 tht0, f64 phi0=0e0):
        TrajFunc(0,0)
    {
        m_kThtSqrtPhi = std::sqrt(2e0);
        m_kRhoSqrtPhi = std::sqrt(2e0)*kRhoTht;
        m_tht0 = tht0;
        m_phi0 = phi0;

        m_p0 = 0e0;
        m_p1 = 1e0/(std::sqrt(8e0)*kRhoTht);
    }

    ~Shell3d_TrajFunc()
    {}

    bool getK(v3* k, f64 p)
    {
        if (k==NULL) return false;
        
        const f64& f64SqrtPhi = p;
        f64 phi = f64SqrtPhi*f64SqrtPhi * (f64SqrtPhi>=0?1e0:-1e0);
        f64 rho = m_kRhoSqrtPhi * f64SqrtPhi;
        f64 tht = m_kThtSqrtPhi * f64SqrtPhi;

        k->x = rho * std::sin(tht+m_tht0) * std::cos(phi+m_phi0);
        k->y = rho * std::sin(tht+m_tht0) * std::sin(phi+m_phi0);
        k->z = rho * std::cos(tht+m_tht0);

        return true;
    }

protected:
    f64 m_kThtSqrtPhi, m_kRhoSqrtPhi;
    f64 m_tht0, m_phi0;
};

class Shell3d: public MrTraj
{
public:
    Shell3d(const GeoPara& objGeoPara, const GradPara& objGradPara, f64 kRhoTht):
        MrTraj(objGeoPara,objGradPara,0,0)
    {
        m_objGeoPara = objGeoPara;
        m_objGradPara = objGradPara;
        m_nRot = calNRot(kRhoTht, m_objGeoPara.nPix);
        m_rotang = calRotAng(m_nRot);
        m_nAcq = m_nRot*m_nRot;
        
        m_vptfBaseTraj.resize(m_nRot);
        m_vvv3BaseGRO.resize(m_nRot);
        m_vv3BaseM0PE.resize(m_nRot);

        m_nSampMax = 0;
        for(i64 i = 0; i < m_nRot; ++i)
        {
            f64 tht0 = i*m_rotang;
            m_vptfBaseTraj[i] = new Shell3d_TrajFunc(kRhoTht, tht0);
            if(!m_vptfBaseTraj[i]) throw std::runtime_error("out of memory");

            calGrad(&m_vv3BaseM0PE[i], &m_vvv3BaseGRO[i], NULL, *m_vptfBaseTraj[i], m_objGradPara);
            m_nSampMax = std::max(m_nSampMax, (i64)m_vvv3BaseGRO[i].size());
        }
    }
    
    virtual ~Shell3d()
    {
        for(i64 i = 0; i < (i64)m_vptfBaseTraj.size(); ++i)
        {
            delete m_vptfBaseTraj[i];
        }
    }

    virtual bool getGRO(vv3* pvv3GRO, i64 iAcq)
    {
        bool ret = true;
        const f64& rotang = m_rotang;
        i64 iSet = iAcq%m_nRot;
        i64 iRot = iAcq/m_nRot;

        *pvv3GRO = m_vvv3BaseGRO[iSet];
        ret &= v3::rotate(pvv3GRO, 2, rotang*iRot, *pvv3GRO);
        
        return ret;
    }

    virtual bool getM0PE(v3* pv3M0PE, i64 iAcq)
    {
        bool ret = true;
        const f64& rotang = m_rotang;
        i64 iSet = iAcq%m_nRot;
        i64 iRot = iAcq/m_nRot;

        *pv3M0PE = m_vv3BaseM0PE[iSet];
        ret &= v3::rotate(pv3M0PE, 2, rotang*iRot, *pv3M0PE);

        return ret;
    }
    
protected:
    i64 m_nRot;
    f64 m_rotang;

    vptf m_vptfBaseTraj;
    vv3 m_vv3BaseM0PE;
    vvv3 m_vvv3BaseGRO;
};
