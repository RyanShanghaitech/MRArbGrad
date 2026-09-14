#pragma once

#include "../utility/global.h"
#include "TrajFunc.h"
#include "../mag/Mag.h"
#include <string>
#include <stdexcept>
#include <ctime>

class ScanPlan
{
public:
    ScanPlan(i64 nPix, i64 nAcqRef, i64 lenReadOut, i64 lenRampFront=0, i64 lenRampBack=0):
        nPix(nPix), nAcqRef(nAcqRef), lenReadOut(lenReadOut), lenRampFront(lenRampFront), lenRampBack(lenRampBack), mag(Mag())
    {}
    virtual ~ScanPlan() {}
    virtual bool getGrad(v3* k0, vv3* grad, v3* k1, i64 iAcq) = 0;
    i64 getNPix() { return nPix; }
    i64 getNAcqRef() { return nAcqRef; }
    i64 getLenRampFront() { return lenRampFront; }
    i64 getLenReadOut() { return lenReadOut; }
    i64 getLenRampBack() { return lenRampBack; }

    static v3 rand(i64 idx);
    static i64 coprime(i64 x);
    static bool permute(vi64* indices, i64 len);

protected:
    i64 nPix, nAcqRef, lenRampFront, lenReadOut, lenRampBack;
    Mag mag;
    
    bool solve(v3* pK0, vv3* pGrad, v3* pK1, vf64* pPara, const TrajFunc& traj);

    static bool extp(vv3* pGrad, vf64* pPara, v3* pM0Front, v3* pM0Back, i64 lenRampFront, i64 lenRampBack, f64 dt);
    static i64 calNRot(const TrajFunc& func, f64 p0, f64 p1, i64 nPix, i64 nSamp=1000);
};

