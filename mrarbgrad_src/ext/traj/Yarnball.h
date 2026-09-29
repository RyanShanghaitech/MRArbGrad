#pragma once

#include "TrajFunc.h"
#include "ScanPlan.h"

class YarnballFunc: public TrajFunc
{
public:
    YarnballFunc(f64 den, f64 tht0, f64 phi0):
        TrajFunc(0,0), tht0(tht0), phi0(phi0)
    {
        /**
         * den: density, phi/rho
         */
        phiDivSqrtTht = std::sqrt(2e0);
        rhoDivSqrtTht = std::sqrt(2e0)/den;

        TrajFunc::p0 = 0e0;
        TrajFunc::p1 = 1e0/(std::sqrt(8e0)/den);
    }

    ~YarnballFunc()
    {}

    virtual bool getK(v3* k, f64 p) const
    {
        if (k==NULL) return false;

        const f64& sqrtTht = p;
        f64 tht = sqrtTht*sqrtTht * (sqrtTht>=0?1e0:-1e0);
        f64 rho = rhoDivSqrtTht * sqrtTht;
        f64 phi = phiDivSqrtTht * sqrtTht;

        k->x = rho * std::sin(tht + tht0) * std::cos(phi + phi0);
        k->y = rho * std::sin(tht + tht0) * std::sin(phi + phi0);
        k->z = rho * std::cos(tht + tht0);

        return true;
    }

protected:
    f64 phiDivSqrtTht, rhoDivSqrtTht;
    f64 tht0, phi0;
};

class YarnballPlan: public ScanPlan
{
public:
    YarnballPlan(i64 nPix, i64 lenRampFront=0, i64 lenRampBack=0, f64 den=2.0*M_PI/0.5):
        ScanPlan(nPix, 0, 0, lenRampFront, lenRampBack), den(den)
    {
        i64 nRot = std::ceil((nPix*M_PI) / (den*0.5));
        nAcqRef = nRot*nRot;
        rot = 2e0*M_PI / nRot;

        YarnballFunc func = YarnballFunc(den, M_PI/2e0, 0);
        mag.setTraj(func);
        vv3 grad; grad.reserve(mag.lenGradRsv);
        mag.solve(&grad, NULL);
        lenReadOut = grad.size();
    }

    virtual bool getGrad(v3* k0, vv3* grad, v3* k1, i64 iAcq)
    {
        bool ret = true;
        iAcq %= nAcqRef;
        
        i64 iPhi = iAcq%nRot, iTht = iAcq/nRot;
        f64 phi0 = rot*iPhi, tht0 = rot*iTht;
        YarnballFunc func(den, tht0, phi0);
        ret &= ScanPlan::solve(k0, grad, k1, NULL, func);
        return ret;
    }

protected:
    f64 den, rot;
    i64 nRot;
};
