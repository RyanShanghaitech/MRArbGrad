#pragma once

#include "TrajFunc.h"
#include "ScanPlan.h"
#include "../utility/LinIntp.h"
#include <algorithm>

class VDSpiralFunc: public TrajFunc
{
public:
    LinIntp intp;
    f64 phi0;

    VDSpiralFunc(const vf64& vRho, const vf64& vDen, f64 phi0):
	TrajFunc(0,0.5), phi0(phi0)
    {
	/**
	 * vRho: radius in k-space
	 * vDenProf: changing rate of phi w.r.t. rho
	 */
	ASSERT(vRho.front()==0.0);
	ASSERT(vRho.back()==0.5);
	intp = LinIntp(vRho, vDen);
    }

    virtual bool getK(v3* k, f64 p) const
    {
	f64& rho = p;
	f64 phi = intp.eval(rho, -1);
        k->x = rho * std::cos(phi + phi0);
        k->y = rho * std::sin(phi + phi0);
        k->z = 0e0;
	return true;
    }
};

class VDSpiralPlan: public ScanPlan
{
private:
    static vf64 vRho_Default()
    {
        vf64 r(2); r[0]=0.0; r[1]=0.5;
        return r;
    }

    static vf64 vDen_Default()
    {
        vf64 d(2);
        d[0]=4.0*M_PI/0.5;
        d[1]=4.0*M_PI/0.5;
        return d;
    }

public:
    VDSpiralPlan(i64 nPix, const vf64& vRho=vRho_Default(), const vf64& vDen=vDen_Default(), i64 lenRampFront=0, i64 lenRampBack=0):
        ScanPlan(nPix, i64(), i64(), lenRampFront, lenRampBack), vRho(vRho), vDen(vDen), func(vRho, vDen, 0.0)
    {
	// rotation angle
	f64 denMin = *std::min_element(vDen.begin(), vDen.end());
	i64 nRot = round((nPix*M_PI) / (denMin*0.5));
        ScanPlan::nAcqRef = nRot;
	rotAng = 2e0*M_PI / (f64)nRot;

	// max readout length
	mag.setTraj(func);
	vv3 grad; mag.solve(&grad, NULL);
        ScanPlan::lenReadOut = grad.size();
    }

    virtual bool getGrad(v3* k0, vv3* grad, v3* k1, i64 iAcq)
    {
	bool ret = true;
	func.phi0 = iAcq*rotAng;
	ret &= ScanPlan::solve(k0, grad, k1, NULL, func);
	return ret;
    }
    
protected:
    vf64 vRho, vDen;
    VDSpiralFunc func;
    f64 rotAng;
};

class SpiralPlan: public VDSpiralPlan
{
private:
    static vf64 getVRho()
    {
        vf64 r(2); r[0]=0.0, r[1]=0.5;
        return r;
    }

    static vf64 getVDen(f64 den)
    {
        vf64 d(2); d[0]=den, d[1]=den;
        return d;
    }

public:
    SpiralPlan(i64 nPix, f64 den=4e0*M_PI/0.5, i64 lenRampFront=0, i64 lenRampBack=0):
        VDSpiralPlan(nPix, getVRho(), getVDen(den), lenRampFront, lenRampBack)
    {}
};

class DDSpiralPlan: public VDSpiralPlan // dual density spiral
{
private:
    static vf64 getVRho(f64 den0, f64 den1, f64 rho0, f64 rho1, i64 nSamp)
    {
        vf64 vRho; vRho.reserve(nSamp+2);
        if (rho0!=0.0) vRho.push_back(0.0);
        for (i64 i=0; i<nSamp; ++i)
        {
            f64 k = (f64)i/(f64)(nSamp-1);
            vRho.push_back(rho0 + (rho1-rho0) * k);
        }
        if (rho1!=0.5) vRho.push_back(0.5);

        return vRho;
    }

    static vf64 getVDen(f64 den0, f64 den1, f64 rho0, f64 rho1, i64 nSamp)
    {
        vf64 vDen; vDen.reserve(nSamp+2);
        if (rho0!=0.0) vDen.push_back(den0);
        for (i64 i=0; i<nSamp; ++i)
        {
            f64 k = (f64)i/(f64)(nSamp-1);
            vDen.push_back(1/(1/den0 + (1/den1-1/den0) * k));
        }
        if (rho1!=0.5) vDen.push_back(den1);

        return vDen;
    }

public:
    DDSpiralPlan(i64 nPix, f64 den0=32.0*M_PI/0.5, f64 den1=2.0*M_PI/0.5, f64 rho0=1/16.0, f64 rho1=0.5, i64 lenRampFront=0, i64 lenRampBack=0):
        VDSpiralPlan(nPix, getVRho(den0, den1, rho0, rho1, nPix), getVDen(den0, den1, rho0, rho1, nPix), lenRampFront, lenRampBack)
    {}
};

