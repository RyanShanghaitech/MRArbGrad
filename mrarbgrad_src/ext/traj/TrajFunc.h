#pragma once

#include "../utility/global.h"
#include "../utility/v3.h"

/* 
 * Single trajectory define by a parameterized function getK()
 * and parameter bounding m_dP0, m_dP1
 */

class TrajFunc
{
public:
    TrajFunc(const f64& p0, const f64& p1):
        p0(p0), p1(p1)
    {}

    virtual ~TrajFunc()
    {}
    
    virtual bool getK(v3* k, f64 p) const = 0; // trajectory function
    
    virtual bool getDkDp(v3* dkdp, f64 p) const // 1st-ord differentiative of trajectory function
    {
        bool ret = true;
        static const f64 dp = 1e-7;
        v3 kNext; ret &= getK(&kNext, p+dp);
        v3 kPrev; ret &= getK(&kPrev, p-dp);
        *dkdp = (kNext-kPrev)/(2e0*dp);

        return ret;
    }

    virtual bool getD2kDp2(v3* d2kdp2, f64 p) const // 2nd-ord differentiative of trajectory function
    {
        static const f64 dp = 1e-3;
        bool ret = true;
        v3 kNext; ret &= getK(&kNext, p+dp);
        v3 kThis; ret &= getK(&kThis, p);
        v3 kPrev; ret &= getK(&kPrev, p-dp);
        *d2kdp2 = (kNext-kThis*2e0+kPrev)/(dp*dp);
    
        return ret;
    }

    // get the lower bound of traj. para.
    f64 getP0() const
    { return p0; }

    // get the upper bound of traj. para.
    f64 getP1() const
    { return p1; }

    bool getK0(v3* k0) const
    { return getK(k0, p0); }
    
    bool getK1(v3* k1) const
    { return getK(k1, p1); }

    // convinient interface
    v3 getK(f64 p) const
    {
        v3 k; getK(&k, p);
        return k;
    }

    v3 getDkDp(f64 p) const
    {
        v3 dkdp; getDkDp(&dkdp, p);
        return dkdp;
    }

    v3 getD2kDp2(f64 p) const
    {
        v3 d2kdp2; getD2kDp2(&d2kdp2, p);
        return d2kdp2;
    }

    v3 getK0() const
    {
        v3 k0; getK0(&k0);
        return k0;
    }

    v3 getK1() const
    {
        v3 k1; getK1(&k1);
        return k1;
    }
    
protected:
    f64 p0, p1;
};
