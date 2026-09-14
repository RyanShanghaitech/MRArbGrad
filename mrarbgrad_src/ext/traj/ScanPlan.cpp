#include "ScanPlan.h"

// a deterministic random number generator
v3 ScanPlan::rand(i64 idx)
{
    v3 ret = true;

    ret.x = std::fmod(idx/(1e0+M_SQRT2), 1e0);

    ret.y = std::fmod(idx/(1e0+M_SQRT3), 1e0);
    ret.y = std::fmod(idx*ret.y, 1e0);

    ret.z = std::fmod(idx/(1e0+M_SQRT7), 1e0);
    ret.z = std::fmod(idx*ret.z, 1e0);
    ret.z = std::fmod(idx*ret.z, 1e0);

    return ret;
}

i64 ScanPlan::coprime(i64 x)
{
    i64 y = (i64)round(x*(GOLDRAT-1));
    i64 inc = 1;
    while (gcd(x,y)!=1)
    {
	y += inc;
	inc = -inc/abs(inc) * (std::abs(inc) + 1);
    }
    return y;
}

// a deterministic shuffle sequence generator
bool ScanPlan::permute(vi64* indices, i64 len)
{
    indices->clear();
    indices->reserve(len);
    i64 inc = coprime(len);

    for(i64 i = 0; i < len; ++i)
    { indices->push_back(i*inc%len); }

    return true;
}

bool ScanPlan::solve(v3* pK0, vv3* pGrad, v3* pK1, vf64* pPara, const TrajFunc& func)
{
    bool ret = true;
    
    // solve for GRO
    ret &= mag.setTraj(func);
    ret &= mag.solve(pGrad, pPara);

    // extrapolate
    v3 m0Front(0), m0Back(0);
    ret &= extp(pGrad, pPara, &m0Front, &m0Back, lenRampFront, lenRampBack, Mag::dt);
    
    // derive k0, k1
    v3 k0, k1;
    k0 = func.getK0() - m0Front;
    v3 m0Grad = pGrad==NULL ? v3(0) : Mag::calM0(*pGrad, Mag::dt);
    k1 = k0 + m0Grad;

    if (pK0) *pK0 = k0;
    if (pK1) *pK1 = k1;

    return ret;
}

bool ScanPlan::extp(vv3* pGrad, vf64* pPara, v3* pM0Front, v3* pM0Back, i64 lenRampFront, i64 lenRampBack, f64 dt)
{
    bool ret = true;

    if (pGrad)
    {
	vv3 ramp;

        // add ramp gradient to zero-start
        ramp = v3::linspace(v3(0), pGrad->front(), lenRampFront+1, true);
        if (pGrad) pGrad->insert(pGrad->begin(), ramp.begin(), std::prev(ramp.end()));
	if (pM0Front) *pM0Front = Mag::calM0(ramp, dt);

        // corresponding parameter sequence
        if (pPara && !pPara->empty())
        { pPara->insert(pPara->begin(), lenRampFront, pPara->front()); }

        // add ramp gradient to ensure zero-end
        ramp = v3::linspace(v3(0), pGrad->back(), lenRampBack+1, true);
        if (pGrad) pGrad->insert(pGrad->end(), std::next(ramp.rbegin()), ramp.rend());
	if (pM0Back) *pM0Back = Mag::calM0(ramp, dt);
        
        // corresponding parameter sequence
        if (pPara && !pPara->empty())
        { pPara->insert(pPara->end(), lenRampBack, pPara->back()); }
    }

    return ret;
}

// calculate required num. of rot. to satisfy Nyquist sampling
i64 ScanPlan::calNRot(const TrajFunc& func, f64 p0, f64 p1, i64 nPix, i64 nSamp)
{
    /*
    * Note:
    * This method is base on the derivative of trajectory function,
    * only local sampling is considered, so there are limitations for
    * this method
    * 
    * Applicable Trajectories:
    * Spiral, Cones, Rosette (single petal)
    * 
    * Non-Applicable Trajectories:
    * Yarnball, Rosette (multi petal)
    */
    // calculate and find min. rot. ang.
    f64 nyq = 1e0/nPix;
    f64 minRotang = 2e0*M_PI;
    for (i64 iK = 1; iK < nSamp-1; ++iK)
    {
        f64 p = p0 + ((f64)iK/nSamp)*(p1-p0);
        v3 k; func.getK(&k, p);
        v3 dkdp;
        {
            v3 v3K_Nx; func.getK(&v3K_Nx, p+1e-7);
            dkdp = v3K_Nx - k;
        }
        if (v3::norm(dkdp)==0) continue;
        v3 dkdphi;
        {
            v3 v3K_Nx; v3::rotate(&v3K_Nx, 2, 1e-7, k);
            dkdphi = v3K_Nx - k;
        }
        if (v3::norm(dkdphi)==0) continue;
        f64 rho = std::sqrt(k.x*k.x + k.y*k.y);
        f64 cosine = v3::inner(dkdp, dkdphi) / (v3::norm(dkdp) * v3::norm(dkdphi));
        f64 sine = std::sqrt(1e0 - std::min(cosine*cosine,1e0));
        f64 rotang = (nyq/sine) / (rho);
        minRotang = std::min(minRotang, rotang);
    }

    // ensure the rot. Num. is an integer
    return (i64)std::ceil(2e0*M_PI/minRotang);
}

