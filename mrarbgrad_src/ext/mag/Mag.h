#pragma once

#include <vector>
#include <list>
#include <tuple>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "../utility/global.h"
#include "../utility/v3.h"
#include "../traj/TrajFunc.h"
#include "../traj/SplineFunc.h"
#include "../utility/LinIntp.h"

class Mag
{
public:
    static f64 dt; static i64 ovsp;
    static f64 sLim, gLim, g0Norm, g1Norm;
    static bool enTrajRep, enGradRep;
    static i64 lenGradRsv, lenTrajRsv;

    Mag();
    bool setTraj(const TrajFunc& func);
    bool setTraj(const vv3& samp);
    bool solve(vv3* grad, vf64* para=NULL);
    template <typename dtype, typename xv3>
    static bool decomp
    (
        std::vector<dtype>* gx,
        std::vector<dtype>* gy,
        std::vector<dtype>* gz,
        const xv3& grad
    );
    static v3 calM0(const vv3& grad, f64 dt);

private:
    // reserved vector for faster computation
    SplineFunc spTrajFunc;
    const TrajFunc* trajFuncPtr;
    vv3 trajSamp;

    vf64 paraBwd;
    vv3 gradBwd;
    vf64 gradBwdNorm;
    vf64 paraFwd;
    vv3 gradFwd;
    vf64 para;
    vv3 grad;

    LinIntp intp;

    f64 curvature(f64 p);
    f64 getDp(const v3& gPrev, const v3& gThis, f64 dt, f64 pPrev, f64 pThis, f64 signDp);
    bool step(v3* gUnit, f64* gNormMin, f64* gNormMax, f64 p, f64 signDp, const v3& g, f64 sLim, f64 dt);
    static bool sovQDE(f64* x0, f64* x1, f64 a, f64 b, f64 c);
};

// definition must be in `.h` file (compiler limitation)
template <typename dtype, typename xv3>
bool Mag::decomp
(
    std::vector<dtype>* pGx,
    std::vector<dtype>* pGy,
    std::vector<dtype>* pGz,
    const xv3& grad
)
{
    bool ret = true;

    std::fill(pGx->begin(), pGx->end(), (dtype)0);
    std::fill(pGy->begin(), pGy->end(), (dtype)0);
    std::fill(pGz->begin(), pGz->end(), (dtype)0);
    
    typename std::vector<dtype>::iterator iGx = pGx->begin();
    typename std::vector<dtype>::iterator iGy = pGy->begin();
    typename std::vector<dtype>::iterator iGz = pGz->begin();
    typename xv3::const_iterator iGrad = grad.begin();
    while (iGrad != grad.end())
    {
        *(iGx++) = dtype(iGrad->x);
        *(iGy++) = dtype(iGrad->y);
        *(iGz++) = dtype(iGrad->z);
        ++iGrad;
    }
    return ret;
}

