#include <cassert>
#include <algorithm>
#include <cstdio>
#include "../utility/global.h"
#include "Mag.h"

f64 Mag::dt = 10e-6; i64 Mag::ovsp = 4; // raster time, oversamp ratio
f64 Mag::sLim = 50 * 42.5756e6 * 1e-3; // slew. limit (Hz/px/s)
f64 Mag::gLim = 30e-3 * 42.5756e6 * 1e-3; // grad. limit (Hz/px)
f64 Mag::g0Norm = 0.0, Mag::g1Norm = 0.0; // start and stop grad. amp.
bool Mag::enTrajRep = true; // trajectory reparameterization
bool Mag::enGradRep = true; // gradient reparameterization
i64 Mag::lenTrajRsv = int(1e4), Mag::lenGradRsv = int(1e5);

Mag::Mag()
{
    // for solver
    this->paraBwd.reserve(Mag::lenGradRsv);
    this->gradBwd.reserve(Mag::lenGradRsv);
    this->gradBwdNorm.reserve(Mag::lenGradRsv);

    this->paraFwd.reserve(Mag::lenGradRsv);
    this->gradFwd.reserve(Mag::lenGradRsv);

    this->para.reserve(Mag::lenGradRsv/Mag::ovsp);
    this->grad.reserve(Mag::lenGradRsv/Mag::ovsp);

    this->intp.init(Mag::lenGradRsv);
    this->intp.eSearchMode = Intp::ECached;

    // for trajectory reparameterization
    this->trajSamp.resize(Mag::lenTrajRsv);
}

bool Mag::setTraj(const TrajFunc& func)
{
    bool ret = true;
    if (Mag::enTrajRep)
    {
        f64 p0 = func.getP0();
        f64 p1 = func.getP1();
	i64 n = this->trajSamp.size();
        for (i64 i = 0; i < n; ++i)
        {
            f64 p = p0 + (p1-p0) * (i)/f64(n-1);
            ret &= func.getK(&this->trajSamp[i], p);
        }
        this->spTrajFunc = SplineFunc(this->trajSamp);
        this->trajFuncPtr = &this->spTrajFunc;
    }
    else
    {
        this->trajFuncPtr = &func;
    }

    return ret;
}

bool Mag::setTraj(const vv3& samp)
{
    this->spTrajFunc = SplineFunc(samp);
    this->trajFuncPtr = &this->spTrajFunc;

    return true;
}

bool Mag::sovQDE(f64* x0, f64* x1, f64 a, f64 b, f64 c)
{
    f64 delta = b*b - 4e0*a*c;
    if (x0) *x0 = (-b-(delta<0?0:std::sqrt(delta)))/(2*a);
    if (x1) *x1 = (-b+(delta<0?0:std::sqrt(delta)))/(2*a);
    return delta>=0;
}

f64 Mag::curvature(f64 p)
{
#define EPS (1e-15)
    v3 dkdp; this->trajFuncPtr->getDkDp(&dkdp, p);
    v3 d2kdp2; this->trajFuncPtr->getD2kDp2(&d2kdp2, p);
    f64 nume = v3::norm(v3::cross(dkdp, d2kdp2));
    f64 deno = pow(v3::norm(dkdp), 3e0);
    if (nume<EPS) nume = EPS;
    if (deno<EPS) deno = EPS;
    return nume/deno;
#undef EPS
}

#if 1

f64 Mag::getDp(const v3& gPrev, const v3& gThis, f64 dt, f64 pPrev, f64 pThis, f64 signDp)
{
    // solve `ΔP` by RK2
    f64 l = v3::norm(gThis)*dt;
    // k1
    f64 k1;
    {
        v3 dkdp; this->trajFuncPtr->getDkDp(&dkdp, pThis);
        f64 dldp = v3::norm(dkdp)*signDp;
        k1 = 1e0/dldp;
    }
    // k2
    f64 k2;
    {
        v3 dkdp; this->trajFuncPtr->getDkDp(&dkdp, pThis+k1*l);
        f64 dldp = v3::norm(dkdp)*signDp;
        k2 = 1e0/dldp;
    }
    f64 dp = l*(0.5*k1 + 0.5*k2);
    return dp;
}

#else // less accurate due to estimation of PNext

f64 Mag::getDp(const v3& v3GPrev, const v3& v3GThis, f64 dt, f64 pPrev, f64 pThis, f64 signDp)
{
    // solve `ΔP` by RK2
    f64 l = v3::norm(v3GThis)*dt;
    v3 dkdp0; ptfTraj->getDkDp(&dkdp0, pThis);
    v3 dkdp1; ptfTraj->getDkDp(&dkdp1, pThis*2e0-pPrev);
    f64 dldp0 = v3::norm(dkdp0)*signDp;
    f64 dldp1 = v3::norm(dkdp1)*signDp;
    return l*(1e0/dldp0 + 1e0/dldp1)/2e0;
}

#endif

bool Mag::step(v3* gUnit, f64* gNormMin, f64* gNormMax, f64 p, f64 signDp, const v3& g, f64 sLim, f64 dt)
{
    // current gradient direction
    v3 dkdp; this->trajFuncPtr->getDkDp(&dkdp, p);
    f64 dldp = v3::norm(dkdp)*signDp;
    if (gUnit) *gUnit = dkdp/dldp;
    
    // current gradient magnitude
    bool isQDESucc = sovQDE
    (
        gNormMin, gNormMax,
        1e0,
        -2e0*v3::inner(g, *gUnit),
        v3::inner(g, g) - std::pow(sLim*dt, 2e0)
    );
    if (gNormMin) *gNormMin = fabs(*gNormMin);
    if (gNormMax) *gNormMax = fabs(*gNormMax);

    return isQDESucc;
}

bool Mag::solve(vv3* grad, vf64* para)
{
    bool ret = true;
    f64 p0 = this->trajFuncPtr->getP0();
    f64 p1 = this->trajFuncPtr->getP1();
    f64 tStep = Mag::dt / Mag::ovsp;
    bool isQDESucc = true; (void)isQDESucc;

    // backward
    v3 g1Unit; ret &= this->trajFuncPtr->getDkDp(&g1Unit, p1);
    g1Unit = g1Unit * (p0>p1?1e0:-1e0);
    g1Unit = g1Unit / v3::norm(g1Unit);
    f64 g1Norm = Mag::g1Norm;
    g1Norm = std::min(g1Norm, Mag::gLim);
    g1Norm = std::min(g1Norm, std::sqrt(Mag::sLim/curvature(p1)));
    v3 g1 = g1Unit * g1Norm;

    this->paraBwd.clear(); this->paraBwd.push_back(p1);
    this->gradBwd.clear(); this->gradBwd.push_back(g1);
    this->gradBwdNorm.clear(); this->gradBwdNorm.push_back(v3::norm(g1));
    while (1)
    {
        f64 p = this->paraBwd.back();
        v3 g = this->gradBwd.back();
        
        // update grad
        v3 gUnit; f64 gNorm;
        isQDESucc = step(&gUnit, NULL, &gNorm, p, (p0-p1)/fabs(p0-p1), g, Mag::sLim, tStep);
        gNorm = std::min(gNorm, Mag::gLim);
        gNorm = std::min(gNorm, std::sqrt(Mag::sLim/curvature(p)));
        g = gUnit*gNorm;

        // update para
        p += getDp(this->gradBwd.back(), g, tStep, this->paraBwd.back(), p, (p0-p1)/fabs(p0-p1));

        // stop or append
        if (fabs(p - p1) >= (1-1e-6)*fabs(p0 - p1))
        {
            // printf("bac: dP/dP1 = %lf/%lf\n", dP, dP1); // test
            break;
        }
        else
        {
            // printf("bac: dP = %lf\n", dP); // test
            this->paraBwd.push_back(p);
            this->gradBwd.push_back(g);
            this->gradBwdNorm.push_back(v3::norm(g));
        }
    }

    std::reverse(this->paraBwd.begin(), this->paraBwd.end());
    std::reverse(this->gradBwdNorm.begin(), this->gradBwdNorm.end());

    this->intp.fit(this->paraBwd, this->gradBwdNorm);
    
    // forward
    v3 g0Unit; ret &= this->trajFuncPtr->getDkDp(&g0Unit, p0);
    g0Unit = g0Unit * (p1>p0?1e0:-1e0);
    g0Unit = g0Unit / v3::norm(g0Unit);
    f64 g0Norm = Mag::g0Norm;
    g0Norm = std::min(g0Norm, Mag::gLim);
    g0Norm = std::min(g0Norm, std::sqrt(Mag::sLim/curvature(p0)));
    g0Norm = std::min(g0Norm, this->intp.eval(p0));
    v3 g0 = g0Unit * g0Norm;

    this->paraFwd.clear(); this->paraFwd.push_back(p0);
    this->gradFwd.clear(); this->gradFwd.push_back(g0);
    while (1)
    {
        f64 p = this->paraFwd.back();
        v3 g = this->gradFwd.back();

        // update grad
        v3 gUnit; f64 gNorm;
        isQDESucc = step(&gUnit, NULL, &gNorm, p, (p1-p0)/fabs(p1-p0), g, Mag::sLim, tStep);
            
	f64 gNormBwd = this->intp.eval(p);
        gNorm = std::min(gNorm, gNormBwd);
        g = gUnit*gNorm;

        // update para
        p += getDp(this->gradFwd.back(), g, tStep, this->paraFwd.back(), p, (p1-p0)/fabs(p1-p0));

        // stop or append
        if (fabs(p - p0) >= (1-1e-6)*fabs(p1 - p0)) // || dGNorm <= 0)
        {
            // printf("for: dP/dP1 = %lf/%lf\n", dP, dP1); // test
            break;
        }
        else
        {
            // printf("for: dP = %lf\n", dP); // test
            this->paraFwd.push_back(p);
            this->gradFwd.push_back(g);
        }
    }
    
    // deoversamp the para. vec.
    if (!para) para = &this->para;
    if (!grad) grad = &this->grad;
    {
        i64 n = this->paraFwd.size();
        para->clear(); para->reserve(n/Mag::ovsp+1);
        grad->clear(); grad->reserve(n/Mag::ovsp+1);
        for (i64 i = 0; i < n; ++i)
        {
            if (i%Mag::ovsp==Mag::ovsp/2)
	    {
		para->push_back(this->paraFwd[i]);
		grad->push_back(this->gradFwd[i]);
	    }
        }
    }

    // derive gradient
    if (Mag::enGradRep)
    {
        v3 k1, k0; 
        vf64::iterator ivf64P = std::next(para->begin());
	vf64::iterator ivf64PPrev = para->begin();
        i64 n = para->size();
        grad->clear();
        for (i64 i = 1; i < n; ++i)
        {
            ret &= this->trajFuncPtr->getK(&k1, *ivf64P);
            ret &= this->trajFuncPtr->getK(&k0, *ivf64PPrev);
            grad->push_back((k1 - k0)/Mag::dt);
	    *ivf64PPrev = *ivf64P; // ensure the consistent length
            ++ivf64P, ++ivf64PPrev;
        }
	para->resize(n-1);
    }

    // extrapolate
    i64 nExtp; vv3 piece;

    nExtp = (i64)std::ceil(v3::norm(grad->front() - g0) / (Mag::sLim*Mag::dt));
    piece = v3::linspace(g0, grad->front(), nExtp, false);
    grad->insert(grad->begin(), piece.begin(), piece.end());
    para->insert(para->begin(), nExtp, para->front());

    nExtp = (i64)std::ceil(v3::norm(grad->back() - g1) / (Mag::sLim*Mag::dt));
    piece = v3::linspace(g1, grad->back(), nExtp, false);
    grad->insert(grad->end(), piece.rbegin(), piece.rend());
    para->insert(para->end(), nExtp, para->back());

    return ret;
}

v3 Mag::calM0(const vv3& grad, f64 dt)
{
    v3 m0 = v3(0,0,0);
    for (int64_t i = 1; i < (i64)grad.size(); ++i)
    { m0 += (grad[i] + grad[i-1]) * dt/2e0; }

    return m0;
}

