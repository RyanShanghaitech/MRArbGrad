#pragma once

#include "TrajFunc.h"
#include "MrTraj.h"
#include <vector>
#include <list>

static bool cvtXyz2Ang(f64* ptht, f64* pphi, const v3& xyz)
{
    const f64& x = xyz.x;
    const f64& y = xyz.y;
    const f64& z = xyz.z;
    f64 xy = std::sqrt(x*x + y*y);
    *ptht = std::atan2(xy, z);
    *pphi = std::atan2(y, x);

    return true;
}

class Seiffert_Trajfunc: public TrajFunc
{
public:
    Seiffert_Trajfunc(f64 m, f64 uMax):
        TrajFunc(0,0)
    {
        m_m = m;
        m_uMax = uMax;

        initJacElip(m_m);

        m_p0 = 0e0;
        m_p1 = uMax;
        m_thtBias = 0e0; m_phiBias = 0e0;
        v3 k1; getK(&k1, uMax);
        cvtXyz2Ang(&m_thtBias, &m_phiBias, k1);
    }
    
    bool getK(v3* k, f64 u)
    {
        if (k==NULL) return false;
        
        f64 sn, cn;
        calJacElip(&sn, &cn, u);

        f64 rho = 0.5e0 * (u/m_uMax);
        k->x = rho * sn * std::cos(u*std::sqrt(m_m));
        k->y = rho * sn * std::sin(u*std::sqrt(m_m));
        k->z = rho * cn;

        v3::rotate(k, 2, -m_phiBias, *k);
        v3::rotate(k, 1, -m_thtBias, *k);

        return true;
    }
    
protected:
    f64 m_m, m_uMax;

    // precompute for AGM
    lf64 m_lf64a, m_lf64b, m_lf64c; 
    
    f64 m_thtBias, m_phiBias;

    bool initJacElip(f64 m)
    {
        if (m<0e0 || m>1e0)
        {
            printf("ArgError, m=%lf\n", m);
            abort();
        }

        // calculate a, b, c value of AGM
        m_lf64a.clear(); m_lf64a.push_back(1e0);
        m_lf64b.clear(); m_lf64b.push_back(std::sqrt(1e0-m));
        m_lf64c.clear(); m_lf64c.push_back(0e0);
        while (std::fabs(*m_lf64b.rbegin() - *m_lf64a.rbegin()) > 1e-8)
        {
            const f64& a_Old = *std::prev(m_lf64a.end());
            const f64& b_Old = *std::prev(m_lf64b.end());
            m_lf64a.push_back((a_Old + b_Old) / 2e0);
            m_lf64b.push_back(std::sqrt(a_Old * b_Old));
            m_lf64c.push_back((a_Old - b_Old) / 2e0);
        }

        return true;
    }
    
    bool calJacElip(f64* psn, f64* pcn, f64 u) const
    {
        // calculate phi with AGM
        i64 n = m_lf64a.size() - 1;
        lf64::const_reverse_iterator ilf64a = m_lf64a.rbegin();
        lf64::const_reverse_iterator ilf64c = m_lf64c.rbegin();
        f64 phi = std::pow(2e0,f64(n)) * (*ilf64a) * u;
        for (i64 i = 0; i < n; ++i)
        {
            f64 _ = (*ilf64c)/(*ilf64a)*std::sin(phi);
            _ = std::max(-1.0, std::min(1.0, _));
            phi = (1e0/2e0)*(phi + std::asin(_));
            ++ilf64a;
            ++ilf64c;
        }
        *psn = std::sin(phi);
        *pcn = std::cos(phi);
        
        return true;
    }
};

class Seiffert: public MrTraj
{
public:
    Seiffert(const GeoPara& objGeoPara, const GradPara& objGradPara, f64 dM, f64 dUMax):
        MrTraj(objGeoPara,objGradPara,0,0)
    // m = 0.07 is optimized for diaphony
    // Umax = 20 can achieve similar readout time as original paper
    {
        const i64& nPix = m_objGeoPara.nPix;
        m_nAcq = (i64)round(-2.53819233e-03*nPix*nPix + 8.53447761e+01*nPix); // fitted

        m_ptfBaseTraj = new Seiffert_Trajfunc(dM, dUMax);
        if(!m_ptfBaseTraj) throw std::runtime_error("out of memory");

        calGrad(&m_v3BaseM0PE, &m_vv3BaseGRO, NULL, *m_ptfBaseTraj, m_objGradPara);
        m_nSampMax = m_vv3BaseGRO.size();
    }
    
    virtual ~Seiffert()
    {
        delete m_ptfBaseTraj;
    }
    
    virtual bool getGRO(vv3* pvv3GRO, i64 iAcq)
    {
        bool ret = true;
        vi64 vi64Ax; vf64 vf64Ang;
        ret &= getRotAng(&vi64Ax, &vf64Ang, iAcq);
        ret &= appRotAng(pvv3GRO, m_vv3BaseGRO, vi64Ax, vf64Ang);

        return ret;
    }

    virtual bool getM0PE(v3* pv3M0PE, i64 iAcq)
    {
        bool ret = true;
        vi64 vi64Ax; vf64 vf64Ang;
        ret &= getRotAng(&vi64Ax, &vf64Ang, iAcq);
        ret &= appRotAng(pv3M0PE, m_v3BaseM0PE, vi64Ax, vf64Ang);

        return ret;
    }
    
protected:
    TrajFunc* m_ptfBaseTraj;
    vv3 m_vv3BaseGRO;
    v3 m_v3BaseM0PE;

    bool getRotAng(vi64* pvi64Ax, vf64* pvf64Ang, i64 iAcq) const
    {
        pvi64Ax->resize(3);
        pvf64Ang->resize(3);

        // randomly rotate around z-axis
        pvi64Ax->at(0) = 2;
        pvf64Ang->at(0) = iAcq*(iAcq+1)*GOLDANG;

        // rotate endpoint to Fibonaci Points
        v3 v3FibPt;
        {
            i64 lNf = m_nAcq;
            f64 dK = f64(iAcq%m_nAcq) - lNf/2;

            f64 dSf = dK/(lNf/2);
            f64 dCf = std::sqrt((lNf/2+dK)*(lNf/2-dK)) / (lNf/2);
            f64 phi = (1e0+std::sqrt(5e0)) / 2e0;
            f64 dTht = 2e0*M_PI*dK/phi;

            v3FibPt.x = dCf*std::sin(dTht);
            v3FibPt.y = dCf*std::cos(dTht);
            v3FibPt.z = dSf;
        }
        f64 tht, phi; cvtXyz2Ang(&tht, &phi, v3FibPt);

        pvi64Ax->at(1) = 1;
        pvf64Ang->at(1) = tht;
        pvi64Ax->at(2) = 2;
        pvf64Ang->at(2) = phi;

        return true;
    }

    template<typename T>
    bool appRotAng(T* pdst, const T& src, vi64 vi64Ax, vf64 vf64Ang) const
    {
        bool ret = true;

        if (vi64Ax.size() != vf64Ang.size()) throw std::invalid_argument("vi64Ax.size() != vf64Ang.size()");

        *pdst = src;
        for(i64 i = 0; i < (i64)vi64Ax.size(); ++i)
        {
            ret &= v3::rotate(pdst, vi64Ax[i], vf64Ang[i], *pdst);
        }

        return ret;
    }
};
