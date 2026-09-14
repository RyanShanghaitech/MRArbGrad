#pragma once

#include "Intp.h"
#include <exception>

class LinIntp : public Intp
{
public:
    LinIntp(i64 sizCache=0)
    {
        if (sizCache) init(sizCache);
    }

    LinIntp(const vf64& vf64X, const vf64& vf64Y)
    {
        init(vf64X.size());
        fit(vf64X, vf64Y);
    }

    bool init(i64 sizCache)
    {
        if (sizCache<=0) return false;
        
        vSlope.reserve(sizCache-1);
        xs.reserve(sizCache);
        ys.reserve(sizCache);

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
        vSlope.resize(nSamp-1);
        vCumSum.resize(nSamp-1);
        for (i64 i=0; i < nSamp-1; ++i)
        {
            const f64 dx = xs[i+1] - xs[i];
            vSlope[i] = (ys[i+1] - ys[i]) / dx;
            if (i!=0)
	    {
		f64 inc = (ys[i]+ys[i-1]) * (xs[i]-xs[i-1]) / 2e0;
		vCumSum[i] = vCumSum[i-1] + inc;
	    }
            else vCumSum[i] = 0e0;
        }

        return true;
    }

    virtual f64 eval(f64 x, i64 ord = 0) const
    {
        ASSERT(xs.size() >= 2);

        const i64 idx = Intp::floor(x);
        const f64 dx = x - xs[idx];

        if (ord == 0)
        { return ys[idx] + vSlope[idx] * dx; }
        else if (ord == 1)
        { return vSlope[idx]; }
        else if (ord == -1)
        { return vCumSum[idx] + (ys[idx]+eval(x,0))*dx/2; }
        else throw std::runtime_error("ord");
    }

private:
    vf64 vSlope, vCumSum;
};
