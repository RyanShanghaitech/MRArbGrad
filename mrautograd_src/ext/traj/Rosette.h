#pragma once

#include "TrajFunc.h"
#include "MrTraj_2D.h"

class Rosette_TrajFunc: public TrajFunc
{
public:
    Rosette_TrajFunc(double dOm1, double dOm2, double dTmax=1e0)
    {
        /*
         * NOTE:
         * When Tmax=1, Om1=Npi, Om2=(N-2)pi,
         * there will be N petal because Om1
         * controls how fast the rho changes.
         */
        m_dOm1 = dOm1;
        m_dOm2 = dOm2;
        m_dTmax = dTmax;

        m_dP0 = 0e0;
        m_dP1 = m_dTmax;
    }

    bool getK(v3* pv3K, double dP) const
    {
        double& dT = dP;
        double dRho = 0.5e0*std::sin(m_dOm1*dT);
        pv3K->m_dX = dRho * std::cos(m_dOm2*dT);
        pv3K->m_dY = dRho * std::sin(m_dOm2*dT);
        pv3K->m_dZ = 0e0;

        return true;
    }
protected:
    double m_dOm1, m_dOm2, m_dTmax;
};

class Rosette: public MrTraj_2D
{
public:
    Rosette(const GeoPara& sGeoPara, const GradPara& sGradPara, double dOm1, double dOm2, double dTmax)
    {
        m_sGeoPara = sGeoPara;
        m_sGradPara = sGradPara;

        m_ptfBasicTraj = new Rosette_TrajFunc(dOm1, dOm2, dTmax);
        if(!m_ptfBasicTraj) throw std::runtime_error("out of memory");
        m_lNRot = calNRot(m_ptfBasicTraj, 0e0, 1e0/(dOm1/M_PI), m_sGeoPara.lNPix);
        m_lNStack = m_sGeoPara.bIs3D ? m_sGeoPara.lNPix : 1;
        m_lNAcq = m_lNRot*m_lNStack;

        m_dRotAngInc = calRotAngInc(m_lNRot);

        calGrad(&m_vv3BasicGrad, *m_ptfBasicTraj, m_sGradPara, 16);
    }
    
    virtual ~Rosette()
    {
        delete m_ptfBasicTraj;
    }
};

class Rosette_Trad: public MrTraj_2D
{
public:
    Rosette_Trad(const GeoPara& sGeoPara, const GradPara& sGradPara, double dOm1, double dOm2, double dTmax)
    {
        m_sGeoPara = sGeoPara;
        m_sGradPara = sGradPara;
        const double& dSLim = m_sGradPara.dSLim;
        const double& dDt = m_sGradPara.dDt;

        m_ptfBasicTraj = new Rosette_TrajFunc(dOm1, dOm2, dTmax);
        if(!m_ptfBasicTraj) throw std::runtime_error("out of memory");
        m_lNRot = calNRot(m_ptfBasicTraj, 0e0, 1e0/(dOm1/M_PI), m_sGeoPara.lNPix);
        m_lNStack = m_sGeoPara.bIs3D ? m_sGeoPara.lNPix : 1;
        m_lNAcq = m_lNRot*m_lNStack;

        m_dRotAngInc = calRotAngInc(m_lNRot);

        // readout
        double dTacq = dTmax/std::sqrt(dSLim*2/(dOm1*dOm1 + dOm2*dOm2));
        int64_t lNSamp = dTacq/dDt;
        vv3 vv3Readout(lNSamp);
        for(int64_t i = 0; i < lNSamp; ++i)
        {
            m_ptfBasicTraj->getDkDp(&vv3Readout[i], dTmax*i/(double)lNSamp); // derivative to p
            vv3Readout[i] = vv3Readout[i]*dTmax/dTacq; // derivative to t
        }

        // pre-ramp
        vv3 vv3Rampup;
        {
            const v3& v3G0 = *vv3Readout.begin();
            double dTRamp = v3::norm(v3G0)/dSLim;
            rampup(&vv3Rampup, v3G0, dTRamp, dDt);
        }
        lNSamp_Rampup = vv3Rampup.size();

        // post-ramp
        vv3 vv3Rampdn;
        {
            const v3& v3G1 = *vv3Readout.rbegin();
            double dTRamp = v3::norm(v3G1)/dSLim;
            rampdn(&vv3Rampdn, v3G1, dTRamp, dDt);
        }
        lNSamp_Rampdn = vv3Rampdn.size();

        // full grad
        lvv3 lvv3GradList;
        lvv3GradList.push_back(vv3Rampup);
        lvv3GradList.push_back(vv3Readout);
        lvv3GradList.push_back(vv3Rampdn);
        concat(&m_vv3BasicGrad, lvv3GradList);
    }
    
    virtual ~Rosette_Trad()
    {
        delete m_ptfBasicTraj;
    }

public:
    int64_t lNSamp_Rampup;
    int64_t lNSamp_Rampdn;
};
