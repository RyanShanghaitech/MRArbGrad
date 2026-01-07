#pragma once

#include "TrajFunc.h"
#include "MrTraj_2D.h"

class Spiral_TrajFunc: public TrajFunc
{
public:
    Spiral_TrajFunc(f64 kRhoPhi):
        TrajFunc(0,0)
    {
        m_kRhoPhi = kRhoPhi;

        m_p0 = 0e0;
        m_p1 = 0.5e0/m_kRhoPhi;
    }

    bool getK(v3* k, f64 p)
    {
        if (k==NULL) return false;
        
        f64& phi = p;
        f64 rho = m_kRhoPhi*phi;
        k->x = rho * std::cos(phi);
        k->y = rho * std::sin(phi);
        k->z = 0e0;

        return true;
    }

protected:
    f64 m_kRhoPhi;
};

class Spiral: public MrTraj_2D
{
public:
    Spiral(const GeoPara& objGeoPara, const GradPara& objGradPara, i64 nStack, f64 kRhoPhi):
        MrTraj_2D(objGeoPara,objGradPara,0,0,0,0,v3(),vv3())
    {
        m_ptfBaseTraj = new Spiral_TrajFunc(kRhoPhi);
        if(!m_ptfBaseTraj) throw std::runtime_error("out of memory");
        m_nStack = nStack;

        i64 nRot = calNRot(kRhoPhi, m_objGeoPara.nPix);
        m_rotang = calRotAng(nRot);
        m_nAcq = nRot*m_nStack;

        calGrad(&m_v3BaseM0PE, &m_vv3BaseGRO, NULL, *m_ptfBaseTraj, m_objGradPara);
        m_nSampMax = m_vv3BaseGRO.size();
    }
    
    virtual ~Spiral()
    {
        delete m_ptfBaseTraj;
    }

protected:
    TrajFunc* m_ptfBaseTraj;
};
