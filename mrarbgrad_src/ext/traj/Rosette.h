#pragma once

#include "TrajFunc.h"
#include "ScanPlan.h"

class RosetteFunc: public TrajFunc
{
public:
    RosetteFunc(f64 om1, f64 om2, f64 tMax, f64 phi0):
        TrajFunc(0,tMax), om1(om1), om2(om2), tMax(tMax), phi0(phi0)
    {
        /*
         * NOTE:
         * When tMax=1, om1=Npi, om2=(N-2)pi,
         * there will be N petal because om1
         * controls how fast the rho changes.
         */
    }

    virtual bool getK(v3* k, f64 p) const
    {
        if (k==NULL) return false;
        
        f64& t = p;
        f64 rho = 0.5e0 * std::sin(om1*t);
        k->x = rho * std::cos(om2*t+phi0);
        k->y = rho * std::sin(om2*t+phi0);
        k->z = 0e0;

        return true;
    }

    virtual bool getDkDp(v3* k, f64 p) const
    {
        if (k==NULL) return false;
        
        f64& t = p;
        f64 rho = 0.5e0 * std::sin(om1*t);
        f64 drho = 0.5e0 * om1 * std::cos(om1*t);
        f64 cos_phi = std::cos(om2*t+phi0);
        f64 dcos_phi = -om2 * std::sin(om2*t+phi0);
        f64 sin_phi = std::sin(om2*t+phi0);
        f64 dsin_phi = om2 * std::cos(om2*t+phi0);
        k->x = rho * dcos_phi + drho * cos_phi;
        k->y = rho * dsin_phi + drho * sin_phi;
        k->z = 0e0;

        return true;
    }

    using TrajFunc::getDkDp;

protected:
    f64 om1, om2, tMax, phi0;
};

class RosettePlan: public ScanPlan
{
public:
    RosettePlan(i64 nPix, i64 lenRampFront=0, i64 lenRampBack=0, f64 om1=5e0*M_PI, f64 om2=3e0*M_PI, f64 tMax=1e0):
        ScanPlan(nPix, 0, 0, lenRampFront, lenRampBack), om1(om1), om2(om2), tMax(tMax)
    {
        // derive nAcqRef, rot
        RosetteFunc func = RosetteFunc(om1, om2, tMax, 0e0);
        i64 nRot = calNRot(func, 0e0, (M_PI/2e0)/om1, nPix);
        nAcqRef = nRot;
	rotAng = 2e0*M_PI / (f64)nRot;

        // max readout length
        mag.setTraj(func);
        vv3 grad; mag.solve(&grad, NULL);
        lenReadOut = grad.size();
    }

    virtual bool getGrad(v3* k0, vv3* grad, v3* k1, i64 iAcq)
    {
        bool ret = true;
        f64 phi0 = iAcq*rotAng;
        RosetteFunc func = RosetteFunc(om1, om2, tMax, phi0);
        ret &= ScanPlan::solve(k0, grad, k1, NULL, func);
        return ret;
    }
    
protected:
    f64 om1, om2, tMax;
    f64 rotAng;
};

class RosetteClassicPlan: public RosettePlan
{
public:
    RosetteClassicPlan(i64 nPix, i64 lenRampFront=0, i64 lenRampBack=0, f64 om1=5e0*M_PI, f64 om2=3e0*M_PI, f64 tMax=1e0, f64 dTE=2e-3):
        RosettePlan(nPix, lenRampFront, lenRampBack, om1, om2, tMax)
    {
        f64 nPetal = tMax / (M_PI/om1);
        tAcq = dTE * nPetal;
        lenReadOut = round(tAcq/Mag::dt);
    }

    virtual bool getGrad(v3* k0, vv3* grad, v3* k1, i64 iAcq)
    {
	bool ret = true;
	f64 om1 = this->om1 * tMax/tAcq;
	f64 om2 = this->om2 * tMax/tAcq;
        f64 phi0 = iAcq*rotAng;

	// derive gradient waveform
        RosetteFunc func = RosetteFunc(om1, om2, tAcq, phi0);
        grad->clear();
	i64 n = round(tAcq/Mag::dt) + 1;
        for (i64 i = 0; i < n; ++i)
        {
            f64 t = i * Mag::dt;
            v3 g = func.getDkDp(t);
            grad->push_back(g);
        }

	// extrapolate
	ret &= ScanPlan::extp(grad, NULL, k0, k1, lenRampFront, lenRampBack, Mag::dt);
	if (k0) *k0 = func.getK0() - *k0;
	if (k1) *k1 = func.getK1() + *k1;

	return ret;
    }
    
protected:
    f64 tAcq;
};

