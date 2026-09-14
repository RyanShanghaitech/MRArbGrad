#pragma once

#include "TrajFunc.h"
#include "ScanPlan.h"

class ConesFunc: public TrajFunc
{
public:
    ConesFunc(f64 den, f64 tht0, f64 phi0):
        TrajFunc(0,0), den(den), tht0(tht0), phi0(phi0)
    {
	TrajFunc::p0 = 0.0;
	TrajFunc::p1 = den*0.5;
    }

    virtual bool getK(v3* k, f64 p) const
    {
        if (k==NULL) return false;
        f64& phi = p;
        f64 rho = phi/den;

        k->x = rho * std::sin(tht0) * std::cos(phi0 + phi);
        k->y = rho * std::sin(tht0) * std::sin(phi0 + phi);
        k->z = rho * std::cos(tht0);

        return true;
    }

protected:
    f64 den, tht0, phi0;
};

class ConesPlan: public ScanPlan
{
public:
    ConesPlan(i64 nPix, i64 lenRampFront=0, i64 lenRampBack=0, f64 den=(4e0*M_PI)/0.5):
        ScanPlan(nPix, 0, 0, lenRampFront, lenRampBack), den(den)
    {
	// derive nAcqRef
	i64 nLayer = (i64)std::ceil(nPix*M_PI/2e0);
	vnRot = vi64(nLayer);
	nAcqRef = 0;
	for (i64 iLayer = 0; iLayer < nLayer; ++iLayer)
	{
	    f64 tht0 = M_PI * (iLayer+0.5)/nLayer;
	    ConesFunc func = ConesFunc(den, tht0, 0e0);
	    vnRot[iLayer] = ScanPlan::calNRot(func, func.getP0(), func.getP1(), nPix);
	    nAcqRef += vnRot[iLayer];
	}

	// derive lenReadOut
	ConesFunc func = ConesFunc(den, M_PI/2e0, 0e0);
	mag.setTraj(func);
	vv3 grad; grad.reserve(mag.lenGradRsv);
	mag.solve(&grad, NULL);
	lenReadOut = grad.size();
    }
    
    virtual bool getGrad(v3* k0, vv3* grad, v3* k1, i64 iAcq)
    {
	bool ret = true;
	iAcq %= nAcqRef;
	i64 nAcqCum = 0;
	i64 iLayer, nLayer = vnRot.size();
	for (iLayer = 0; iLayer < nLayer; ++iLayer)
	{
	    if (nAcqCum+vnRot[iLayer] > iAcq) break;
	    else nAcqCum += vnRot[iLayer];
	}
	i64 iRot = iAcq - nAcqCum;
	f64 tht0 = M_PI * (iLayer+0.5)/nLayer;
	f64 phi0 = 2*M_PI * iRot / vnRot[iLayer];
	ConesFunc func = ConesFunc(den, tht0, phi0);
	ret &= ScanPlan::solve(k0, grad, k1, NULL, func);
	return ret;
    }

protected:
    f64 den;
    vi64 vnRot;
};
