#pragma once

#include "Intp.h"

class SplineIntp : public Intp
{
public:
    SplineIntp(i64 sizCache=0)
    {
        if (sizCache) init(sizCache);
    }

    SplineIntp(const vf64& xs, const vf64& ys)
    {
        init(xs.size());
        fit(xs, ys);
    }

    bool init(i64 sizCache)
    {
        if (sizCache<=0) return false;

        xs.reserve(sizCache);
        ys.reserve(sizCache);

        vH.reserve(sizCache-1);
        vAlpha.reserve(sizCache);
        vL.reserve(sizCache);
        vMu.reserve(sizCache);
        vZ.reserve(sizCache);

        vA.reserve(sizCache);
        vB.reserve(sizCache-1);
        vC.reserve(sizCache);
        vD.reserve(sizCache-1);
        
        return true;
    }

    virtual bool fit(const vf64& xs, const vf64& ys)
    {
        ASSERT(xs.size() == ys.size());
        const i64 nSamp = xs.size();
        ASSERT(nSamp >= 2);

        idxCache = 0;

        this->xs = xs;
        this->ys = ys;

        vH.resize(nSamp-1);
        vAlpha.resize(nSamp);
        vL.resize(nSamp);
        vMu.resize(nSamp);
        vZ.resize(nSamp);

        vA = ys;
        vB.resize(nSamp-1);
        vC.resize(nSamp);
        vD.resize(nSamp-1);

        vL[0]  = 1.0;
        vMu[0] = 0.0;
        vZ[0]  = 0.0;
        vAlpha[0] = 0.0;
        vAlpha[nSamp-1] = 0.0;

        for (i64 i = 0; i < nSamp-1; ++i)
        {
            vH[i] = xs[i+1] - xs[i];
        }

        // Step 1: Set up the tridiagonal system
        for (i64 i = 1; i < nSamp-1; ++i)
            vAlpha[i] = (3e0 / vH[i]) * (ys[i+1] - ys[i]) - (3e0 / vH[i-1]) * (ys[i] - ys[i-1]);

        // Step 2: Solve tridiagonal system for c (second derivatives)
        for (i64 i = 1; i < nSamp-1; ++i)
        {
            vL[i] = 2e0 * (xs[i+1] - xs[i-1]) - vH[i-1] * vMu[i-1];
            vMu[i] = vH[i] / vL[i];
            vZ[i] = (vAlpha[i] - vH[i-1] * vZ[i-1]) / vL[i];
        }

        // Natural spline boundary conditions
        vL[nSamp-1] = 1.0;
        vZ[nSamp-1] = 0.0;
        vC[nSamp-1] = 0.0;

        // Back substitution
        for (i64 i=nSamp-2; i>=0; --i)
        {
            vC[i] = vZ[i] - vMu[i] * vC[i+1];
            vB[i] = (vA[i+1] - vA[i]) / vH[i] - vH[i] * (vC[i+1] + 2e0 * vC[i]) / 3e0;
            vD[i] = (vC[i+1] - vC[i]) / (3e0 * vH[i]);
        }

        return true;
    }

    virtual f64 eval(f64 x, i64 ord=0) const
    {
        i64 idx = Intp::floor(x);

        f64 dx = x - xs[idx];
        f64 dx2 = dx * dx;
        f64 dx3 = dx2 * dx;
        if (ord == 0) return
        (
            vA[idx]
            + vB[idx] * dx
            + vC[idx] * dx2
            + vD[idx] * dx3
        );
        if (ord == 1) return
        (
            vB[idx]
            + vC[idx] * 2e0 * dx
            + vD[idx] * 3e0 * dx2
        );
        if (ord == 2) return
        (
            vC[idx] * 2e0
            + vD[idx] * 6e0 * dx
        );
        if (ord == 3) return
        (
            vD[idx] * 6e0
        );
        return 0e0;
    }

private:
    vf64 vH, vAlpha, vL;
    vf64 vMu, vZ;
    vf64 vA, vB, vC, vD;
};
