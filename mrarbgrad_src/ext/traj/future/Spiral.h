#pragma once

#include "TrajFunc.h"
#include "MrTraj_2D.h"

class Spiral_TrajFunc: public TrajFunc
{
public:
    Spiral_TrajFunc(f64 kRhoPhi, f64 phi0):
        TrajFunc(0,0),
	kRhoPhi(kRhoPhi),
	phi0(phi0)
    {
        this->p0 = 0e0;
        this->p1 = 0.5e0/kRhoPhi;
    }

    bool getK(v3* k, f64 p) const
    {
        if (k==NULL) return false;
        
        f64& phi = p;
        f64 rho = kRhoPhi*phi;
        k->x = rho * std::cos(phi+phi0);
        k->y = rho * std::sin(phi+phi0);
        k->z = 0e0;

        return true;
    }

protected:
    f64 kRhoPhi, phi0;
};

class Spiral: public MrTraj_2D
{
public:
    Spiral(const GeoPara& objGeoPara, const GradPara& objGradPara, i64 nStack, f64 kRhoPhi):
        MrTraj_2D(objGeoPara,objGradPara,0,0,0,0,v3(),vv3())
    {
        m_ptfBaseTraj = new Spiral_TrajFunc(kRhoPhi);
        ASSERT(m_ptfBaseTraj!=NULL);
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
