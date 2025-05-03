#pragma once

#include "TrajFunc.h"
#include "../core/GradGen.h"
#include <string>
#include <stdexcept>
#include <random>

#define GOLDRAT ((1e0+std::sqrt(5e0))/2e0)
#define GOLDANG ((3e0-std::sqrt(5e0))*M_PI)

/* 
 * A set of trajectories sufficient to fully-sample the k-space
 * defined by:
 * 1. some basic trajectories with different shapes.
 * 2. a acquisition plan which decides which basic trajectory to use,
 *    and how to transform to the desired gradient of a particular acquisition.
 * 
 * notice:
 * 1. Different basic trajectoires share the same traj. func. getK(),
 *    but the behaviour of getK() may differ due to constant parameters.
 */

class MrTraj
{
public:
    typedef std::vector<int64_t> vl;
    typedef std::list<int64_t> ll;
    typedef std::vector<double> vd;
    typedef std::string str;
    typedef std::vector<v3> vv3;
    typedef std::vector<vv3> vvv3;
    typedef std::list<vv3> lvv3;
    typedef std::vector<TrajFunc*> vptf;
    typedef struct
    {
        bool bIs3D;
        double dFov;
        int64_t lNPix;
    } GeoPara;
    typedef struct
    {
        double dSLim;
        double dGLim;
        double dDt;
    } GradPara;
    
    const double dGyoMagRat; // Hz/T
    
    MrTraj():
        dGyoMagRat(42.5756e6)
    {}
    
    virtual ~MrTraj()
    {}
    
    virtual bool getGrad(v3* pv3K0, vv3* pvv3Grad, int64_t lIAcq) const = 0;

    bool is3D() const
    { return m_sGeoPara.bIs3D; }

    double getFov() const
    { return m_sGeoPara.dFov; }

    int64_t getNPix() const
    { return m_sGeoPara.lNPix; }

    double getSLim() const
    { return m_sGradPara.dSLim; }

    double getGLim() const
    { return m_sGradPara.dGLim; }

    double getDt() const
    { return m_sGradPara.dDt; }
    
    int64_t getNAcq() const
    { return m_lNAcq; }

    static bool revGrad(v3* pv3M0Dst, vv3* pvv3Dst, const v3& v3M0Src, const vv3& vv3Src, double dDt)
    {
        if(vv3Src.size() <= 1) throw std::invalid_argument("vv3Src.size()");

        // derive K1
        *pv3M0Dst = v3M0Src;
        for(int64_t i = 1; i < (int64_t)vv3Src.size(); ++i)
        {
            *pv3M0Dst = *pv3M0Dst + (vv3Src[i] + vv3Src[i-1])*dDt/2e0;
        }

        // reverse gradient
        *pvv3Dst = vv3(vv3Src.rbegin(), vv3Src.rend());
        for(int64_t i = 0; i < (int64_t)pvv3Dst->size(); ++i)
        {
            pvv3Dst->at(i).m_dX = -pvv3Dst->at(i).m_dX;
            pvv3Dst->at(i).m_dY = -pvv3Dst->at(i).m_dY;
            pvv3Dst->at(i).m_dZ = -pvv3Dst->at(i).m_dZ;
        }

        return true;
    }

    static bool rampup(vv3* pvv3GradDst, const v3& v3G0, double dTRamp, double dDt)
    {
        int64_t lNSamp_Ramp = (int64_t)std::ceil(dTRamp/dDt);

        *pvv3GradDst = vv3(lNSamp_Ramp*4);
        for(int64_t i = 0; i < lNSamp_Ramp; ++i)
        {
            pvv3GradDst->at(i) = v3G0 * -i/(double)lNSamp_Ramp;
        }
        for(int64_t i = 0; i < lNSamp_Ramp; ++i)
        {
            pvv3GradDst->at(i+lNSamp_Ramp) = v3G0 * -(lNSamp_Ramp-i)/(double)lNSamp_Ramp;
        }
        for(int64_t i = 0; i < 2*lNSamp_Ramp; ++i)
        {
            pvv3GradDst->at(i+2*lNSamp_Ramp) = v3G0 * i/(double)(2*lNSamp_Ramp);
        }

        return true;
    }

    static bool rampdn(vv3* pvv3GradDst, const v3& v3DkDt1, double dTRamp, double dDt)
    {
        int64_t lNSamp_Ramp = (int64_t)std::ceil(dTRamp/dDt);
        
        *pvv3GradDst = vv3(lNSamp_Ramp);
        for(int64_t i = 1; i <= lNSamp_Ramp; ++i)
        {
            pvv3GradDst->at(i-1) = v3DkDt1 * (lNSamp_Ramp-i)/(double)lNSamp_Ramp;
        }

        return true;
    }

    static bool concat(vv3* pvv3Grad, const lvv3& lvv3GradList)
    {
        int64_t lNSamp = 0;
        lvv3::const_iterator ilvv3;

        ilvv3 = lvv3GradList.begin();
        while(ilvv3 != lvv3GradList.end())
        {
            lNSamp += ilvv3->size();
            ++ilvv3;
        }

        pvv3Grad->clear();
        pvv3Grad->reserve(lNSamp);

        ilvv3 = lvv3GradList.begin();
        while(ilvv3 != lvv3GradList.end())
        {
            pvv3Grad->insert(pvv3Grad->end(), ilvv3->begin(), ilvv3->end());
            ++ilvv3;
        }

        return true;
    }

    static bool calGrad(vv3* pvv3G, const TrajFunc& tf, const GradPara& sGradPara, int64_t lOs=10)
    {
        bool bRet = true;
        const double& dSLim = sGradPara.dSLim;
        const double& dGLim = sGradPara.dGLim;
        const double& dDt = sGradPara.dDt;

        // calculate gradient
        GradGen gg(&tf, dSLim, dGLim, dDt, lOs, 0e0, 0e0, true);
        bRet &= gg.compute(pvv3G);

        return true;
    }

    static bool genRandIdx(vl* pvlIdx, int64_t lN)
    {
        ll llSeq;
        for(int64_t i = 0; i < lN; ++i)
        {
            llSeq.push_back(i);
        }
        ll::iterator illSeq = llSeq.begin();
        pvlIdx->resize(lN);
        
        int64_t lIntv = (int64_t)round(llSeq.size()*(llSeq.size()+1)*(1e0/(GOLDRAT+1e0))); lIntv %= std::max(llSeq.size(), (size_t)1);
        for(int64_t i = 0; i < lN; ++i)
        {
            for(int64_t j = 0; j < lIntv; ++j)
            {
                illSeq++;
                if(illSeq==llSeq.end()) illSeq=llSeq.begin();
            }
            pvlIdx->at(i) = *illSeq;
            illSeq = llSeq.erase(illSeq);
            if(illSeq==llSeq.end()) illSeq=llSeq.begin();
            lIntv = (int64_t)round(llSeq.size()*(llSeq.size()+1)*(1e0/(GOLDRAT+1e0))); lIntv %= std::max(llSeq.size(), (size_t)1);
        }

        return true;
    }

protected:
    GeoPara m_sGeoPara;
    GradPara m_sGradPara;
    int64_t m_lNAcq;
    
    // calculate required num. of rot. to satisfy Nyquist sampling (for spiral only)
    static int64_t calNRot(double dRhoRotAng, int64_t lNPix)
    {
        return (int64_t)std::ceil(lNPix*2e0*M_PI*dRhoRotAng);
    }

    // calculate required num. of rot. to satisfy Nyquist sampling
    static int64_t calNRot(TrajFunc* ptraj, double dP0, double dP1, int64_t lNPix, int64_t lNSamp=1000)
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
        double dNyqIntv = 1e0/lNPix;
        double dMinRotAng = 2e0*M_PI;
        for (int64_t lIk = 1; lIk < lNSamp-1; ++lIk)
        {
            double dP = dP0 + ((double)lIk/lNSamp)*(dP1-dP0);
            v3 v3K; ptraj->getK(&v3K, dP);
            v3 v3DkDpara;
            {
                v3 v3K_Nx; ptraj->getK(&v3K_Nx, dP+1e-7);
                v3DkDpara = v3K_Nx - v3K;
            }
            if (v3::norm(v3DkDpara)==0) continue;
            v3 v3DkDphi;
            {
                v3 v3K_Nx; v3::rotate(&v3K_Nx, 2, 1e-7, v3K);
                v3DkDphi = v3K_Nx - v3K;
            }
            if (v3::norm(v3DkDphi)==0) continue;
            double dRho = std::sqrt(v3K.m_dX*v3K.m_dX + v3K.m_dY*v3K.m_dY);
            double dCos = v3::inner(v3DkDpara, v3DkDphi) / (v3::norm(v3DkDpara) * v3::norm(v3DkDphi));
            double dSin = std::sqrt(1e0 - std::min(dCos*dCos,1e0));
            double dRotAng = (dNyqIntv/dSin) / (dRho);
            dMinRotAng = std::min(dMinRotAng, dRotAng);
        }

        // ensure the rot. Num. is a integer
        return (int64_t)std::ceil(2e0*M_PI/dMinRotAng);
    }
    
    static double calRotAngInc(int64_t lNRot)
    {
        return 2e0*M_PI/lNRot;
    }
};
