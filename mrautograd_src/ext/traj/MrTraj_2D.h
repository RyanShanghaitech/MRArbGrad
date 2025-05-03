#pragma once

#include "MrTraj.h"
#include "../core/GradGen.h"

class MrTraj_2D: public MrTraj
{
public:
    MrTraj_2D() {}
    
    virtual ~MrTraj_2D()
    {}
    
    bool getGrad(v3* pv3K0, vv3* pgGrad, int64_t lIAcq) const
    {
        bool bRet = true;
        int64_t lIStack = lIAcq%m_lNStack;
        int64_t lIRot = lIAcq/m_lNStack;
        
        m_ptfBasicTraj->getK0(pv3K0);
        pv3K0->m_dZ += getK0z(lIStack, m_lNStack);

        bRet &= v3::rotate(pv3K0, 2, m_dRotAngInc*lIRot, *pv3K0);
        bRet &= v3::rotate(pgGrad, 2, m_dRotAngInc*lIRot, m_vv3BasicGrad);

        return bRet;
    }

    int64_t getNRot()
    { return m_lNRot; }

    int64_t getNStack()
    { return m_lNStack; }

    double getRotAngInc()
    { return m_dRotAngInc; }

    bool tran2GoldAng(int64_t lNRot=1000000)
    {
        m_lNRot = lNRot;
        m_dRotAngInc = GOLDANG;
        return true;
    }

protected:
    int64_t m_lNRot, m_lNStack;
    double m_dRotAngInc;
    TrajFunc* m_ptfBasicTraj;
    vv3 m_vv3BasicGrad;

    static double getK0z(int64_t lIStack, int64_t lNStack=256)
    {
        return lIStack/(double)lNStack - (lNStack/2)/(double)lNStack;
    }
};
