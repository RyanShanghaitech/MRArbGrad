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
        m_objGeoPara = objGeoPara;
        m_objGradPara = objGradPara;
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
            if(!m_vptfBaseTraj[i]) throw std::runtime_error("out of memory");

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

/* incomplete - we plan to test 2D real-time first before 3D
class AxrollYarnball_RT: public MrTraj
{
public:
    AxrollYarnball_RT(const GeoPara& objGeoPara, const GradPara& objGradPara, f64 dRhoPhi, i64 lNAcq)
    {
        m_objGeoPara = objGeoPara;
        m_objGradPara = objGradPara;
        m_nAcq = lNAcq;

        m_dRhoPhi = dRhoPhi;

        m_vv3M0PE.resize(lNAcq); std::fill(m_vv3M0PE.begin(), m_vv3M0PE.end(), v3(-1,-1,-1));
        m_vlNWait.resize(lNAcq); std::fill(m_vlNWait.begin(), m_vlNWait.end(), -1);
        m_vlNSamp.resize(lNAcq); std::fill(m_vlNSamp.begin(), m_vlNSamp.end(), -1);
    }

    virtual ~AxrollYarnball_RT()
    {
        ;
    }
    
    virtual bool getGRO(vv3* pvv3GRO, i64 iAcq) const
    {
        bool ret = true;
        v3 v3Tht0Phi0; genRand3d(&v3Tht0Phi0, iAcq);
        v3Tht0Phi0 *= 2*M_PI;
        v3Tht0Phi0 -= M_PI;
        TrajFunc* ptfTraj = new Yarnball_TrajFunc(m_dRhoPhi, v3Tht0Phi0.x, v3Tht0Phi0.y);
        if (!ptfTraj) throw std::runtime_error("out of memory");
        if (iAcq>=m_nAcq) throw std::runtime_error("iAcq>=m_nAcq");
        ret &= calGrad(&m_vv3M0PE[iAcq], plv3GRO, NULL, &m_vlNWait[iAcq], &m_vlNSamp[iAcq], *ptfTraj, m_objGradPara, 4);
        m_vv3M0PE[iAcq] = v3::axisroll(m_vv3M0PE[iAcq], iAcq%3);
        {
            lv3::iterator ilv3GRO = plv3GRO->begin();
            while (ilv3GRO!=plv3GRO->end())
            {
                *ilv3GRO = v3::axisroll(*ilv3GRO, iAcq % 3);
            }
        }
        delete ptfTraj;
        return ret;
    }

    virtual bool getM0PE(v3* pv3M0PE, i64 iAcq) const
    {
        if (iAcq>=m_nAcq)
        {
            throw std::runtime_error("iAcq>=m_nAcq");
        }
        *pv3M0PE = m_vv3M0PE[iAcq];
        return true;
    }

    virtual i64 getNWait(i64 iAcq) const
    {
        if (iAcq>=m_nAcq)
        {
            throw std::runtime_error("iAcq>=m_nAcq");
        }
        return m_vlNWait[iAcq];
    }

    virtual i64 getNSamp(i64 iAcq) const
    {
        if (iAcq>=m_nAcq)
        {
            throw std::runtime_error("iAcq>=m_nAcq");
        }
        return m_vlNSamp[iAcq];
    }

protected:
    f64 m_dRhoPhi;

    vv3 m_vv3M0PE;
    vl m_vlNWait;
    vl m_vlNSamp;
}
*/