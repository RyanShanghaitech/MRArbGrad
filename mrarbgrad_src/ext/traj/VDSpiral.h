#pragma once

#include "TrajFunc.h"
#include "MrTraj_2D.h"

class VDSpiral_TrajFunc: public TrajFunc
{
public:
    VDSpiral_TrajFunc(f64 kRhoPhi0, f64 kRhoPhi1, f64 phi0=0e0):
        TrajFunc(0,0)
    {
        m_kRhoPhi0 = kRhoPhi0;
        m_kRhoPhi1 = kRhoPhi1;
        m_phi0 = phi0;

        m_p0 = 0e0;
        m_p1 = (std::log(m_kRhoPhi1)-std::log(m_kRhoPhi0)) / (2e0*(m_kRhoPhi1-m_kRhoPhi0));
    }

    virtual bool getK(v3* k, f64 p)
    {
        if (k==NULL) return false;
        
        f64& phi = p;
        f64 rho = m_kRhoPhi0*(std::exp(2e0*(m_kRhoPhi1 - m_kRhoPhi0)*phi) - 1e0) / (2e0*(m_kRhoPhi1 - m_kRhoPhi0));
        k->x = rho * std::cos(phi + m_phi0);
        k->y = rho * std::sin(phi + m_phi0);
        k->z = 0e0;

        return true;
    }

protected:
    f64 m_kRhoPhi0;
    f64 m_kRhoPhi1;
    f64 m_phi0;
};

class VDSpiral: public MrTraj_2D
{
public:
    VDSpiral(const GeoPara& objGeoPara, const GradPara& objGradPara, i64 nStack, f64 kRhoPhi0, f64 dRhoPhi1):
        MrTraj_2D(objGeoPara,objGradPara,0,0,0,0,v3(),vv3())
    {
        if (kRhoPhi0==dRhoPhi1) throw std::invalid_argument("kRhoPhi0==dRhoPhi1");

        m_ptfBaseTraj = new VDSpiral_TrajFunc(kRhoPhi0, dRhoPhi1);
        if(!m_ptfBaseTraj) throw std::runtime_error("out of memory");
        m_nStack = nStack;

        i64 nRot = calNRot(std::max(kRhoPhi0, dRhoPhi1), m_objGeoPara.nPix);
        m_rotang = calRotAng(nRot);
        m_nAcq = nRot*m_nStack;

        calGrad(&m_v3BaseM0PE, &m_vv3BaseGRO, NULL, *m_ptfBaseTraj, m_objGradPara);
        m_nSampMax = m_vv3BaseGRO.size();
    }
    
    virtual ~VDSpiral()
    {
        delete m_ptfBaseTraj;
    }

protected:
    TrajFunc* m_ptfBaseTraj;
};

class VDSpiral_RT: public MrTraj
{
    // TODO: Goldang sampling is incomplete, shuffled sampling is incomplete.
public:
    VDSpiral_RT(const GeoPara& objGeoPara, const GradPara& objGradPara, f64 kRhoPhi0, f64 dRhoPhi1, f64 nAcq):
        MrTraj(objGeoPara,objGradPara,0,0)
    {
        m_nAcq = nAcq;

        m_dRhoPhi0 = kRhoPhi0;
        m_dRhoPhi1 = dRhoPhi1;

        m_vv3M0PE.resize(nAcq); std::fill(m_vv3M0PE.begin(), m_vv3M0PE.end(), v3(0));
    }

    virtual ~VDSpiral_RT()
    {}

    virtual bool getGRO(vv3* pvv3GRO, i64 iAcq)
    {
        bool ret = true;
        TrajFunc* ptfTraj = new VDSpiral_TrajFunc(m_dRhoPhi0, m_dRhoPhi1, iAcq*GOLDANG);
        if (!ptfTraj) throw std::runtime_error("out of memory");
        if (iAcq>=m_nAcq) throw std::runtime_error("iAcq>=m_nAcq");
        ret &= calGrad(&m_vv3M0PE[iAcq], pvv3GRO, NULL, *ptfTraj, m_objGradPara, 4);
        delete ptfTraj;
        return ret;
    }

    virtual bool getM0PE(v3* pv3M0PE, i64 iAcq)
    {
        if (iAcq>=m_nAcq)
        {
            throw std::runtime_error("iAcq>=m_nAcq");
        }
        *pv3M0PE = m_vv3M0PE[iAcq];
        return true;
    }

    virtual i64 getNWait(i64 iAcq)
    {
        if (iAcq>=m_nAcq)
        {
            throw std::runtime_error("iAcq>=m_nAcq");
        }
        return m_vlNWait[iAcq];
    }

    virtual i64 getNSamp(i64 iAcq)
    {
        if (iAcq>=m_nAcq)
        {
            throw std::runtime_error("iAcq>=m_nAcq");
        }
        return m_vlNSamp[iAcq];
    }

protected:
    f64 m_dRhoPhi0;
    f64 m_dRhoPhi1;

    vv3 m_vv3M0PE;
    vi64 m_vlNWait;
    vi64 m_vlNSamp;
};