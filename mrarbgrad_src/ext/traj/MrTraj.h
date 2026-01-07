#pragma once

#include "TrajFunc.h"
#include "../mag/Mag.h"
#include <string>
#include <stdexcept>
#include <ctime>

#ifdef USE_MTG
#include "../mtg/header.h"
#endif

bool gMrTraj_enMtg = false; // whether to use Lustig's minTimeGrad solver
f64 gMrTraj_g0Norm = 0e0; // initial gradient amplitude
f64 gMrTraj_g1Norm = 0e0; // final gradient amplitude

/* 
 * A set of trajectories sufficient to fully-sample the k-space
 * defined by:
 * 1. some Base trajectories with different shapes.
 * 2. a acquisition plan which decides which Base trajectory to use,
 *    and how to transform to the desired gradient of a particular acquisition.
 * Or, do nothing in the constructor and compute the gradient on the fly
 * in getGRO() function.
 * 
 * notice:
 * 1. Different Base trajectoires share the same traj. func. getK(),
 *    but the behaviour of getK() may differ due to constant parameters.
 */

class MrTraj
{
public:
    typedef std::vector<TrajFunc*> vptf;
    typedef struct
    {
        f64 fov;
        i64 nPix;
    } GeoPara;
    typedef struct
    {
        f64 sLim;
        f64 gLim;
        f64 dt;
    } GradPara;
    
    MrTraj(const GeoPara& m_objGeoPara, const GradPara& m_objGradPara, const i64& m_nAcq, const i64& m_nSampMax):
        m_gamma(42.5756e6),
        m_objGeoPara(m_objGeoPara),
        m_objGradPara(m_objGradPara),
        m_nAcq(m_nAcq),
        m_nSampMax(m_nSampMax)
    { mag = Mag(); }
    
    virtual ~MrTraj()
    {}
    
    virtual bool getGRO(vv3* pvv3GRO, i64 iAcq) = 0;
    
    virtual bool getM0PE(v3* pv3M0PE, i64 iAcq) = 0;

    const GeoPara& getGeoPara()
    { return m_objGeoPara; }

    const GradPara& getGradPara()
    { return m_objGradPara; }
    
    i64 getNAcq()
    { return m_nAcq; }
    
    i64 getNSampMax()
    { return m_nSampMax; }

    f64 getGyoMagRat()
    { return m_gamma; }

    void setGyoMagRat(f64 x)
    { m_gamma = x; }

    void setUseMtg(bool x)
    { gMrTraj_enMtg = x; }

    void setGNorm0(f64 x)
    { gMrTraj_g0Norm = x; }

    void setGNorm1(f64 x)
    { gMrTraj_g1Norm = x; }

    // a deterministic random number generator
    static bool genRand3d(v3* v3Res, i64 lIdx)
    {
        v3Res->x = fmod(lIdx/(1e0+M_SQRT2), 1e0);

        v3Res->y = fmod(lIdx/(1e0+M_SQRT3), 1e0);
        v3Res->y = fmod(lIdx*v3Res->y, 1e0);

        v3Res->z = fmod(lIdx/(1e0+M_SQRT7), 1e0);
        v3Res->z = fmod(lIdx*v3Res->z, 1e0);
        v3Res->z = fmod(lIdx*v3Res->z, 1e0);

        return true;
    }

    // a deterministic shuffle sequence generator
    static bool genRandIdx(vi64* pvi64Idx, i64 num)
    {
        // resize target container rationally
        pvi64Idx->clear();
        pvi64Idx->reserve(num);

        // generate sequantial index
        vi64 vi64SeqIdx; vi64SeqIdx.reserve(num);
        for(i64 i = 0; i < num; ++i)
        { vi64SeqIdx.push_back(i); }
        
        // decide step size, make step size and num of idx coprime
        i64 inc = (i64)(num*(GOLDRAT-1));
        while (gcd(inc, num)!=1)
        { ++inc; }

        // generate random index
        for(i64 i = 0; i < num; ++i)
        { pvi64Idx->push_back(i*inc%num); }

        return true;
    }

    static bool calM0SP(v3* pv3M0SP, const v3& v3M0PE, const vv3 vv3GRO)
    {
        bool ret = true;
        *pv3M0SP = v3M0PE;
        vv3::const_iterator iterGRO = vv3GRO.begin();
        vv3::const_iterator iterGRO_NonZero = iterGRO;
        for(i64 i = 0; i < (i64)vv3GRO.size()-1; ++i)
        {
            *pv3M0SP += (*iterGRO + *std::next(iterGRO))*1e0/2e0; // assume `dt` to be 1
            if (v3::norm(*iterGRO)!=0e0) iterGRO_NonZero = iterGRO;
            ++iterGRO;
        }

        if (v3::norm(*pv3M0SP) != 0e0)
        { *pv3M0SP /= v3::norm(*pv3M0SP); }
        else if (v3::norm(*iterGRO_NonZero) != 0e0)
        { *pv3M0SP = *iterGRO_NonZero/v3::norm(*iterGRO_NonZero); }
        else
        { *pv3M0SP = v3(1,0,0); }

        return ret;
    }

protected:
    // constant
    f64 m_gamma; // Hz/T

    // trajectory info
    GeoPara m_objGeoPara;
    GradPara m_objGradPara;
    i64 m_nAcq;
    i64 m_nSampMax;

    // solver settings
    Mag mag;
    
    // calculate required num. of rot. to satisfy Nyquist sampling (for spiral only)
    static i64 calNRot(f64 kRhoPhi, i64 nPix)
    {
        return (i64)std::ceil(nPix*2e0*M_PI*kRhoPhi);
    }

    // calculate required num. of rot. to satisfy Nyquist sampling
    static i64 calNRot(TrajFunc* ptraj, f64 p0, f64 p1, i64 nPix, i64 nSamp=1000)
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
    {
        // calculate and find min. rot. ang.
        f64 nyq = 1e0/nPix;
        f64 minRotang = 2e0*M_PI;
        for (i64 iK = 1; iK < nSamp-1; ++iK)
        {
            f64 p = p0 + ((f64)iK/nSamp)*(p1-p0);
            v3 k; ptraj->getK(&k, p);
            v3 dkdp;
            {
                v3 v3K_Nx; ptraj->getK(&v3K_Nx, p+1e-7);
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

        // ensure the rot. Num. is a integer
        return (i64)std::ceil(2e0*M_PI/minRotang);
    }
    
    static f64 calRotAng(i64 nRot)
    {
        return 2e0*M_PI/nRot;
    }

    bool calGRO_MAG(vv3* pvv3G, vf64* pvf64P, TrajFunc& tf, const GradPara& objGradPara, i64 oversamp=8)
    {
        bool ret = true;
        const f64& sLim = objGradPara.sLim;
        const f64& gLim = objGradPara.gLim;
        const f64& dt = objGradPara.dt;

        mag.setup(&tf, sLim, gLim, dt, oversamp, gMrTraj_g0Norm, gMrTraj_g1Norm);
        ret &= mag.solve(pvv3G, pvf64P);

        return ret;
    }
    
    bool calGRO_MAG(vv3* pvv3G, vf64* pvf64P, const vv3& vv3TrajSamp, const GradPara& objGradPara, i64 oversamp=8)
    {
        bool ret = true;
        const f64& sLim = objGradPara.sLim;
        const f64& gLim = objGradPara.gLim;
        const f64& dt = objGradPara.dt;

        mag.setup(vv3TrajSamp, sLim, gLim, dt, oversamp, gMrTraj_g0Norm, gMrTraj_g1Norm);
        ret &= mag.solve(pvv3G, pvf64P);

        return ret;
    }

    bool calGRO_MTG(vv3* pvv3G, vf64* pvf64P, const vf64& vf64C, const GradPara& objGradPara)
    {
        #ifdef USE_MTG
        bool ret = true;
        const f64& sLim = objGradPara.sLim;
        const f64& gLim = objGradPara.gLim;
        const f64& dt = objGradPara.dt;
        if (pvf64P) pvf64P->clear(); // does not supported

        // Prepare arg. for Lustig's function
        f64 g0 = gMrTraj_g0Norm, gfin = gMrTraj_g1Norm, gmax = gLim, smax = sLim, T = dt, ds = -1; // ds = 35e-4 for const-Nstep comparison

        f64 *p_Cx = nullptr, *p_Cy = nullptr, *p_Cz = nullptr;
        f64 *p_gx = nullptr, *p_gy = nullptr, *p_gz = nullptr;
        f64 *p_sx = nullptr, *p_sy = nullptr, *p_sz = nullptr;
        f64 *p_kx = nullptr, *p_ky = nullptr, *p_kz = nullptr;
        f64 *p_sdot = nullptr, *p_sta = nullptr, *p_stb = nullptr;
        
        f64 time = 0;
        int size_interpolated = 0, size_sdot = 0, size_st = 0;
        int gfin_empty = (gfin<0), ds_empty = (ds<0);

        // Call Lustig's function (assume it is linked in or compiled as C)
        minTimeGradientRIV(
            vf64C.data(), vf64C.size()/3, 3, g0, gfin, gmax, smax, T, ds,
            &p_Cx, &p_Cy, &p_Cz, &p_gx, &p_gy, &p_gz,
            &p_sx, &p_sy, &p_sz, &p_kx, &p_ky, &p_kz, &p_sdot, &p_sta, &p_stb, &time,
            &size_interpolated, &size_sdot, &size_st, gfin_empty, ds_empty);

        // Copy results to C++ outputs
        pvv3G->clear();
        for (i64 i = 0; i < size_interpolated; ++i)
        {
            pvv3G->push_back(v3(p_gx[i], p_gy[i], p_gz[i]));
        }
        
        free(p_Cx);    free(p_Cy);    free(p_Cz);
        free(p_gx);    free(p_gy);    free(p_gz);
        free(p_sx);    free(p_sy);    free(p_sz);
        free(p_kx);    free(p_ky);    free(p_kz);
        free(p_sdot);  free(p_sta);   free(p_stb);

        return ret;
        #else

        char sErrMsg[] = "MTG not found";
        puts(sErrMsg);
        throw std::runtime_error(sErrMsg);
        return false;

        #endif
    }

    bool calGRO(vv3* pvv3G, vf64* pvf64P, TrajFunc& tf, GradPara& objGradPara, i64 oversamp=8)
    {
        bool ret = true;
        const i64 nTrajSamp = 1000;

        // calculate gradient
        if(!gMrTraj_enMtg)
        {
            ret &= calGRO_MAG(pvv3G, pvf64P, tf, objGradPara, oversamp);
        }
        else
        {
            // Prepare trajectory sampling
            vf64 vf64C(nTrajSamp * 3, 0.0);

            // Sample the trajectory at N points
            f64 dP0 = tf.getP0();
            f64 dP1 = tf.getP1();
            for (i64 i = 0; i < nTrajSamp; ++i)
            {
                f64 dP = dP0 + (dP1-dP0)* (i)/f64(nTrajSamp-1);
                v3 v3K; tf.getK(&v3K, dP);
                v3K *= 4.257; // k is defined by k*4.257 in Lustig's method
                vf64C[i] = v3K.x;
                vf64C[i + nTrajSamp] = v3K.y;
                vf64C[i + 2*nTrajSamp] = v3K.z;
            }

            ret &= calGRO_MTG(pvv3G, pvf64P, vf64C, objGradPara);
        }

        return ret;
    }

    bool calGRO(vv3* pvv3G, vf64* pvf64P, vv3& vv3TrajSamp, const GradPara& objGradPara, i64 oversamp=8)
    {
        bool ret = true;
        i64 nTrajSamp = vv3TrajSamp.size();

        // calculate gradient
        if(!gMrTraj_enMtg)
        {
            ret &= calGRO_MAG(pvv3G, pvf64P, vv3TrajSamp, objGradPara, oversamp);
        }
        else
        {
            // Prepare trajectory sampling
            vf64 vf64C(nTrajSamp*3);

            // Sample the trajectory at N points
            for (int i = 0; i < nTrajSamp; ++i)
            {
                v3 v3K = vv3TrajSamp[i]*4.257; // k is defined by k*4.257 in Lustig's method
                vf64C[i] = v3K.x;
                vf64C[i + nTrajSamp] = v3K.y;
                vf64C[i + 2*nTrajSamp] = v3K.z;
            }

            ret &= calGRO_MTG(pvv3G, pvf64P, vf64C, objGradPara);
        }

        return ret;
    }

    bool calGrad(v3* pv3M0PE, vv3* pvv3GRO, vf64* pvf64P, TrajFunc& tfTraj, GradPara& objGradPara, i64 oversamp=8)
    {
        bool ret = true;
        const f64& sLim = objGradPara.sLim;
        const f64& dt = objGradPara.dt;
        
        // calculate GRO
        TIC;
        ret &= calGRO(pvv3GRO, pvf64P, tfTraj, objGradPara, oversamp);
        TOC;

        // if GEnd needs to be fixed
        if (gMrTraj_g0Norm==0e0 && pvv3GRO)
        {
            // add ramp gradient to satisfy desired Gstart and Gfinal
            vv3 vv3GRampFront;
            ret &= Mag::ramp_front(&vv3GRampFront, pvv3GRO->front(), v3(), sLim, dt);
            pvv3GRO->insert(pvv3GRO->begin(), vv3GRampFront.begin(), vv3GRampFront.end());
            
            // corresponding parameter sequence
            if (pvf64P && !pvf64P->empty()) // null: user does not need p seq, empty: q seq not supported
            {
                vf64 vf64PInsert = vf64(vv3GRampFront.size(), pvf64P->front());
                pvf64P->insert(pvf64P->begin(), vf64PInsert.begin(), vf64PInsert.end());
            }
        }
        if (gMrTraj_g1Norm==0e0 && pvv3GRO)
        {
            // add ramp gradient to satisfy desired Gstart and Gfinal
            vv3 vv3GRampBack;
            ret &= Mag::ramp_back(&vv3GRampBack, pvv3GRO->back(), v3(), sLim, dt);
            pvv3GRO->insert(pvv3GRO->end(), vv3GRampBack.begin(), vv3GRampBack.end());
            
            // corresponding parameter sequence
            if (pvf64P && !pvf64P->empty()) // null: user does not need p seq, empty: q seq not supported
            {
                vf64 vf64PInsert = vf64(vv3GRampBack.size(), pvf64P->back());
                pvf64P->insert(pvf64P->end(), vf64PInsert.begin(), vf64PInsert.end());
            }
        }

        if (pv3M0PE) ret &= tfTraj.getK0(pv3M0PE);

        return ret;
    }
};
