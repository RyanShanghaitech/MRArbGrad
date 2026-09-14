#pragma once

#include "TrajFunc.h"
#include "ScanPlan.h"
#include "../utility/LinIntp.h"
#include <algorithm>

class VDSpiralFunc: public TrajFunc
{
public:
    VDSpiralFunc(vf64 vRho, vf64 vDenProf, f64 phi0):
	TrajFunc(0,0.5), phi0(phi0)
    {
	/**
	 * vRho: radius in k-space
	 * vDenProf: changing rate of phi w.r.t. rho
	 */
	ASSERT(vRho.front()==0.0);
	ASSERT(vRho.back()==0.5);
	intp = LinIntp(vRho, vDenProf);
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

protected:
    LinIntp intp;
    f64 phi0;
};

class VDSpiralPlan: public ScanPlan
{
private:
    static vf64 vRho_default()
    {
        vf64 r(2);
        r[0]=0.0;
        r[1]=0.5;
        return r;
    }

    static vf64 vDenProf_default()
    {
        vf64 r(2);
        r[0]=4e0*M_PI/0.5;
        r[1]=4e0*M_PI/0.5;
        return r;
    }

public:
    VDSpiralPlan(i64 nPix, i64 lenRampFront=0, i64 lenRampBack=0, const vf64& vRho=vRho_default(), const vf64& vDenProf=vDenProf_default()):
        ScanPlan(nPix, 0, 0, lenRampFront, lenRampBack), vRho(vRho), vDenProf(vDenProf)
    {
	// rotation angle vector
	f64 denMin = *std::min_element(vDenProf.begin(), vDenProf.end());
	i64 nRot = round((nPix*M_PI) / (denMin*0.5));
        nAcqRef = nRot;
	rotAng = 2e0*M_PI / (f64)nRot;

	// max readout length
	VDSpiralFunc func = VDSpiralFunc(vRho, vDenProf, 0e0);
	mag.setTraj(func);
	vv3 grad; mag.solve(&grad, NULL);
	lenReadOut = grad.size();
    }

    virtual bool getGrad(v3* k0, vv3* grad, v3* k1, i64 iAcq)
    {
	bool ret = true;
	f64 phi0 = iAcq*rotAng;
	VDSpiralFunc func = VDSpiralFunc(vRho, vDenProf, phi0);
	ret = ScanPlan::solve(k0, grad, k1, NULL, func);
	return ret;
    }
    
protected:
    vf64 vRho, vDenProf;
    f64 rotAng;
};

class SpiralPlan: public ScanPlan
{
public:
    SpiralPlan(i64 nPix, i64 lenRampFront=0, i64 lenRampBack=0, f64 den=4e0*M_PI/0.5):
        ScanPlan(nPix, 0, 0, lenRampFront, lenRampBack)
    {
	// rotation angle vector
	i64 nRot = round((nPix*M_PI) / (den*0.5));
        nAcqRef = nRot;
	rotAng = 2e0*M_PI / (f64)nRot;

	// max readout length
        vRho.resize(2); vRho[0] = 0; vRho[1] = 0.5;
        vDenProf.resize(2); vDenProf[0] = den; vDenProf[1] = den;
	VDSpiralFunc func = VDSpiralFunc(vRho, vDenProf, 0e0);
	mag.setTraj(func);
	vv3 grad; mag.solve(&grad, NULL);
	lenReadOut = grad.size();
    }

    virtual bool getGrad(v3* k0, vv3* grad, v3* k1, i64 iAcq)
    {
	bool ret = true;
	f64 phi0 = iAcq*rotAng;
	VDSpiralFunc func = VDSpiralFunc(vRho, vDenProf, phi0);
	ret &= ScanPlan::solve(k0, grad, k1, NULL, func);
	return ret;
    }
    
protected:
    vf64 vRho, vDenProf;
    f64 rotAng;
};

class LVDSpiralPlan: public ScanPlan // linear variable density spiral
{
public:
    LVDSpiralPlan(i64 nPix, i64 lenRampFront=0, i64 lenRampBack=0, f64 denIn=256.0*M_PI/0.5, f64 denOt=2.0*M_PI/0.5):
        ScanPlan(nPix, 0, 0, lenRampFront, lenRampBack)
    {
	// rotation angle vector
        f64 denMin = std::min(denIn, denOt);
	i64 nRot = round((nPix*M_PI) / (denMin*0.5));
        nAcqRef = nRot;
	rotAng = 2e0*M_PI / (f64)nRot;

	// max readout length
        vRho.resize(2); vRho[0] = 0; vRho[1] = 0.5;
        vDenProf.resize(2); vDenProf[0] = denIn; vDenProf[1] = denOt;
	VDSpiralFunc func = VDSpiralFunc(vRho, vDenProf, 0e0);
	mag.setTraj(func);
	vv3 grad; mag.solve(&grad, NULL);
	lenReadOut = grad.size();
    }

    virtual bool getGrad(v3* k0, vv3* grad, v3* k1, i64 iAcq)
    {
	bool ret = true;
	f64 phi0 = iAcq*rotAng;
	VDSpiralFunc func = VDSpiralFunc(vRho, vDenProf, phi0);
	ret &= ScanPlan::solve(k0, grad, k1, NULL, func);
	return ret;
    }
    
protected:
    vf64 vRho, vDenProf;
    f64 rotAng;
};

