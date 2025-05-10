#pragma once

#include "TrajFunc.h"
#include "../core/GradGen.h"
#include <string>
#include <stdexcept>
#include <ctime>

#define GOLDRAT ((1e0+std::sqrt(5e0))/2e0)
#define GOLDANG ((3e0-std::sqrt(5e0))*M_PI)

#define TIC \
    clock_t cTick = std::clock();\

#define TOC \
    cTick = std::clock() - cTick;\
    printf("Elapsed time: %.3f ms\n", (float)1e3*cTick/CLOCKS_PER_SEC);

/* 
 * A set of trajectories sufficient to fully-sample the k-space
 * defined by:
 * 1. some Base trajectories with different shapes.
 * 2. a acquisition plan which decides which Base trajectory to use,
 *    and how to transform to the desired gradient of a particular acquisition.
 * 
 * notice:
 * 1. Different Base trajectoires share the same traj. func. getK(),
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
        bool bMaxG0;
        bool bMaxG1;
    } GradPara;
    
    const double dGyoMagRat; // Hz/T
    
    MrTraj():
        dGyoMagRat(42.5756e6)
    {}
    
    virtual ~MrTraj()
    {}
    
    virtual bool getM0PE(v3* pv3M0PE, int64_t lIAcq) const = 0;
    
    virtual bool getGRO(vv3* pvv3GRO, int64_t lIAcq) const = 0;
    
    virtual bool getM0SP(v3* pv3M0SP, int64_t lIAcq) const = 0;
    
    virtual int64_t getNWaitAdc(int64_t lIAcq) const = 0;
    
    virtual int64_t getNSampAdc(int64_t lIAcq) const = 0;

    const GeoPara& getGeoPara() const
    { return m_sGeoPara; }

    const GradPara& getGradPara() const
    { return m_sGradPara; }
    
    int64_t getNAcq() const
    { return m_lNAcq; }

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

    static bool calGRO(vv3* pvv3G, const TrajFunc& tf, const GradPara& sGradPara, int64_t lOs=16)
    {
        bool bRet = true;
        const double& dSLim = sGradPara.dSLim;
        const double& dGLim = sGradPara.dGLim;
        const double& dDt = sGradPara.dDt;
        const bool& bMaxG0 = sGradPara.bMaxG0;
        const bool& bMaxG1 = sGradPara.bMaxG1;

        // calculate gradient
        TIC;

        GradGen gg(&tf, dSLim, dGLim, dDt, lOs, bMaxG0?1e15:0e0, bMaxG1?1e15:0e0);
        bRet &= gg.compute(pvv3G);

        TOC;

        return true;
    }

    static bool calGrad(v3* pv3M0PE, vv3* pvv3GRO, v3* pv3M0SP, int64_t* plNWaitAdc, int64_t* plNSampAdc, const TrajFunc* ptfBaseTraj, const GradPara& sGradPara, int64_t lOs=16, double dTRampFront=0.5e-3, double dTRampBack=0.5e-3)
    {
        bool bRet = true;
        const double& dDt = sGradPara.dDt;
        
        // calculate GRO with ramp-up and ramp-down
        vv3 vv3GRO; calGRO(&vv3GRO, *ptfBaseTraj, sGradPara, lOs);
        vv3 vv3GRampFront; bRet &= GradGen::ramp_front(&vv3GRampFront, *vv3GRO.begin(), v3(0,0,0), int64_t(dTRampFront/dDt), dDt);
        vv3 vv3GRampBack; bRet &= GradGen::ramp_back(&vv3GRampBack, *vv3GRO.rbegin(), v3(0,0,0), int64_t(dTRampBack/dDt), dDt);
        
        lvv3 lvv3FullGRO;
        lvv3FullGRO.push_back(vv3GRampFront);
        lvv3FullGRO.push_back(vv3GRO);
        lvv3FullGRO.push_back(vv3GRampBack);
        bRet &= GradGen::catGrad(pvv3GRO, lvv3FullGRO);
        *plNWaitAdc = vv3GRampFront.size();
        *plNSampAdc = vv3GRO.size();

        // calculate M0 of PE
        bRet &= ptfBaseTraj->getK0(pv3M0PE);
        *pv3M0PE = *pv3M0PE - GradGen::calM0(vv3GRampFront, dDt, v3(0,0,0), *vv3GRO.begin());

        // calculate M0 of SP
        bRet &= ptfBaseTraj->getK1(pv3M0SP);
        *pv3M0SP = *pv3M0SP + GradGen::calM0(vv3GRampBack, dDt, *vv3GRO.rbegin(), v3(0,0,0));
        *pv3M0SP = v3::norm(*pv3M0SP)!=0?*pv3M0SP/v3::norm(*pv3M0SP):v3(1,0,0);

        return bRet;
    }
};
