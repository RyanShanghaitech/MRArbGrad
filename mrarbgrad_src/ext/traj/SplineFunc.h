#pragma once

#include "TrajFunc.h"
#include "../utility/SplineIntp.h"

typedef std::vector<v3> vv3;

class SplineFunc: public TrajFunc
{
public:
    SplineFunc():
        TrajFunc(0,0),
        intpX(0),
        intpY(0),
        intpZ(0)
    {}

    SplineFunc(const vv3& vv3K):
        TrajFunc(0,0),
        intpX(vv3K.size()),
        intpY(vv3K.size()),
        intpZ(vv3K.size())
    {
        i64 nTrajSamp = vv3K.size();

        vf64 vf64P(nTrajSamp);
        vf64P[0] = 0;
        for (i64 i = 1; i < nTrajSamp; ++i)
        {
            vf64P[i] = vf64P[i-1] + v3::norm(vv3K[i] - vv3K[i-1]);
        }

        vf64 vf64X(nTrajSamp), vf64Y(nTrajSamp), vf64Z(nTrajSamp);
        for (i64 i = 0; i < nTrajSamp; ++i)
        {
           vf64X[i] = vv3K[i].x;
           vf64Y[i] = vv3K[i].y;
           vf64Z[i] = vv3K[i].z;
        }

        intpX.eSearchMode = Intp::ECached;
        intpY.eSearchMode = Intp::ECached;
        intpZ.eSearchMode = Intp::ECached;

        intpX.fit(vf64P, vf64X); 
        intpY.fit(vf64P, vf64Y);
        intpZ.fit(vf64P, vf64Z);

        p0 = *vf64P.begin();
        p1 = *vf64P.rbegin();
    }
    
    bool getK(v3* k, f64 p) const
    {
        k->x = intpX.eval(p);
        k->y = intpY.eval(p);
        k->z = intpZ.eval(p);

        return true;
    }

    bool getDkDp(v3* dkdp, f64 p) const
    {
        dkdp->x = intpX.eval(p, 1);
        dkdp->y = intpY.eval(p, 1);
        dkdp->z = intpZ.eval(p, 1);

        return true;
    }

    bool getD2kDp2(v3* d2kdp2, f64 p) const
    {
        d2kdp2->x = intpX.eval(p, 2);
        d2kdp2->y = intpY.eval(p, 2);
        d2kdp2->z = intpZ.eval(p, 2);

        return true;
    }
protected:
    SplineIntp intpX, intpY, intpZ;
};
