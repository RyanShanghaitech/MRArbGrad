#pragma once

#include "TrajFunc.h"
#include "MrTraj.h"

class Yarnball_TrajFunc: public TrajFunc
{
public:
    Yarnball_TrajFunc(f64 kRhoPhi, f64 tht0, f64 phi0=0e0):
        TrajFunc(0,0)
    {
        m_kPhiSqrtTht = std::sqrt(2e0);
        m_kRhoSqrtTht = std::sqrt(2e0)*kRhoPhi;
        m_tht0 = tht0;
        m_phi0 = phi0;

        m_p0 = 0e0;
        m_p1 = 1e0/(std::sqrt(8e0)*kRhoPhi);
    }

    ~Yarnball_TrajFunc()
    {}

    virtual bool getK(v3* k, f64 p)
    {
        if (k==NULL) return false;

        const f64& dSqrtTht = p;
        f64 tht = dSqrtTht*dSqrtTht * (dSqrtTht>=0?1e0:-1e0);
        f64 rho = m_kRhoSqrtTht * dSqrtTht;
        f64 phi = m_kPhiSqrtTht * dSqrtTht;

        k->x = rho * std::sin(tht+m_tht0) * std::cos(phi+m_phi0);
        k->y = rho * std::sin(tht+m_tht0) * std::sin(phi+m_phi0);
        k->z = rho * std::cos(tht+m_tht0);

        return true;
    }

protected:
    f64 m_kPhiSqrtTht, m_kRhoSqrtTht;
    f64 m_tht0, m_phi0;
};

class Yarnball: public MrTraj
{
public:
    Yarnball(const GeoPara& objGeoPara, const GradPara& objGradPara, f64 kRhoPhi):
        MrTraj(objGeoPara,objGradPara,0,0)
    {
        m_nRot = calNRot(kRhoPhi, m_objGeoPara.nPix);
        m_rotang = calRotAng(m_nRot);
        m_nAcq = m_nRot*m_nRot;
        
        m_vptfBaseTraj.resize(m_nRot);
        m_vvv3BaseGRO.resize(m_nRot);
        m_vv3BaseM0PE.resize(m_nRot);

        m_nSampMax = 0;
        for(i64 i = 0; i < m_nRot; ++i)
        {
            f64 tht0 = i*m_rotang;
            m_vptfBaseTraj[i] = new Yarnball_TrajFunc(kRhoPhi, tht0);
            ASSERT(m_vptfBaseTraj[i]!=NULL);

            calGrad(&m_vv3BaseM0PE[i], &m_vvv3BaseGRO[i], NULL, *m_vptfBaseTraj[i], m_objGradPara);
            m_nSampMax = std::max(m_nSampMax, (i64)m_vvv3BaseGRO[i].size());
        }
    }
    
    virtual ~Yarnball()
    {
        for(i64 i = 0; i < (i64)m_vptfBaseTraj.size(); ++i)
        {
            delete m_vptfBaseTraj[i];
        }
    }

    virtual bool getGrad(v3* pv3M0PE, vv3* pvv3GRO, i64 iAcq)
    {
        bool ret = true;
        const f64& rotang = m_rotang;
        i64 iPhi = iAcq%m_nRot;
        i64 iTht = iAcq/m_nRot%m_nRot;

        if (pv3M0PE)
        {
            *pv3M0PE = m_vv3BaseM0PE[iTht];
            ret &= v3::rotate(pv3M0PE, 2, rotang*iPhi, *pv3M0PE);
        }
        if (pvv3GRO)
        {
            *pvv3GRO = m_vvv3BaseGRO[iTht];
            ret &= v3::rotate(pvv3GRO, 2, rotang*iPhi, *pvv3GRO);
        }

        return ret;
    }

protected:
    i64 m_nRot;
    f64 m_rotang;

    vptf m_vptfBaseTraj;
    vv3 m_vv3BaseM0PE;
    vvv3 m_vvv3BaseGRO;
};

class Yarnball_RT: public MrTraj
{
public:
    Yarnball_RT(const GeoPara& objGeoPara, const GradPara& objGradPara, f64 kRhoPhi):
        MrTraj(objGeoPara,objGradPara,0,0)
    {
        m_kRhoPhi = kRhoPhi;
        m_nRot = calNRot(kRhoPhi, objGeoPara.nPix);
        m_dRotAng = 2e0*M_PI/m_nRot;
        m_nAcq = m_nRot*m_nRot;
        genPermTab(&m_vi64PermTab, m_nAcq);
        
        Yarnball_TrajFunc tf(m_kRhoPhi, M_PI/2e0, 0);
        vv3 vv3GRO; calGrad(NULL, &vv3GRO, NULL, tf, m_objGradPara, 4);
        m_nSampMax = vv3GRO.size() + 1; // 1 for redundance
    }

    virtual ~Yarnball_RT()
    {}
    
    virtual bool getGrad(v3* pv3M0PE, vv3* pvv3GRO, i64 iAcq)
    {
        bool ret = true;
        ASSERT(iAcq >= 0);
        i64 iPhi = m_vi64PermTab[iAcq%m_nAcq]%m_nRot;
        i64 iTht = m_vi64PermTab[iAcq%m_nAcq]/m_nRot;
        f64 phi = m_dRotAng*iPhi;
        f64 tht = m_dRotAng*iTht;
        Yarnball_TrajFunc tf(m_kRhoPhi, tht, phi);
        ret &= calGrad(pv3M0PE, pvv3GRO, NULL, tf, m_objGradPara, 2);
        return ret;
    }

protected:
    f64 m_kRhoPhi;
    i64 m_nRot; f64 m_dRotAng;
    vi64 m_vi64PermTab;
};
