#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstdio>
#include <ctime>
#include <algorithm>
#include "core/GradGen.h"
#include "traj/TrajFunc.h"
#include "traj/MrTraj.h"
#include "traj/Spiral.h"
#include "traj/VarDenSpiral.h"
#include "traj/Rosette.h"
#include "traj/Shell3d.h"
#include "traj/Yarnball.h"
#include "traj/Seiffert.h"
#include "traj/Cones.h"

#define FLAG_REVERSE (0)
#define FLAG_GOLDANG (0)
#define FLAG_MAXG0 (0)
#define FLAG_MAXG1 (0)

typedef std::vector<double> vd;
typedef std::vector<int64_t> vl;
typedef std::vector<v3> vv3;
typedef std::vector<vv3> vvv3;

PyObject* cvtVv3toNparr(vv3& vv3Src)
{
    int iD0 = vv3Src.size();
    // allocate numpy array
    PyObject* ppyoG;
    {
        npy_intp aDims[] = {iD0, 3};
        ppyoG = PyArray_ZEROS(2, aDims, NPY_FLOAT32, 0);
    }

    // fill the data in
    for (int64_t i = 0; i < (int)vv3Src.size(); ++i)
    {
        *(float*)PyArray_GETPTR2((PyArrayObject*)ppyoG, i, 0) = vv3Src[i].m_dX;
        *(float*)PyArray_GETPTR2((PyArrayObject*)ppyoG, i, 1) = vv3Src[i].m_dY;
        *(float*)PyArray_GETPTR2((PyArrayObject*)ppyoG, i, 2) = vv3Src[i].m_dZ;
    }

    return ppyoG;
}

PyObject* cvtVvv3toList(vvv3& vvv3Src)
{
    PyObject* ppyoList = PyList_New(0);
    for (int64_t i = 0; i < (int)vvv3Src.size(); ++i)
    {
        PyObject* ppyoArr = cvtVv3toNparr(vvv3Src[i]);
        PyList_Append(ppyoList, ppyoArr);
    }
    return ppyoList;
}

PyObject* cvtV3toNparr(v3& v3Src)
{
    // allocate numpy array
    PyObject* ppyoG;
    {
        npy_intp aDims[] = {3};
        ppyoG = PyArray_ZEROS(1, aDims, NPY_FLOAT32, 0);
    }

    // fill the data in
    *(float*)PyArray_GETPTR1((PyArrayObject*)ppyoG, 0) = v3Src.m_dX;
    *(float*)PyArray_GETPTR1((PyArrayObject*)ppyoG, 1) = v3Src.m_dY;
    *(float*)PyArray_GETPTR1((PyArrayObject*)ppyoG, 2) = v3Src.m_dZ;

    return ppyoG;
}

PyObject* cvtVv3toList(vv3& vv3Src)
{
    PyObject* ppyoList = PyList_New(0);
    for (int64_t i = 0; i < (int)vv3Src.size(); ++i)
    {
        PyObject* ppyoArr = cvtV3toNparr(vv3Src[i]);
        PyList_Append(ppyoList, ppyoArr);
    }
    return ppyoList;
}

bool inline checkNarg(int64_t lNarg, int64_t lNargExp)
{
    if (lNarg != lNargExp)
    {
        printf("wrong num. of arg, narg=%ld, %ld expected\n", lNarg, lNargExp);
        abort();
        return false;
    }
    return true;
}

bool getGeoGradPara(PyObject* const* args, MrTraj::GeoPara* psGeoPara, MrTraj::GradPara* psGradPara)
{
    *psGeoPara = 
    {
        (bool)PyLong_AsLong(args[0]),
        (double)PyFloat_AsDouble(args[1]),
        (int64_t)PyLong_AsLong(args[2])
    };

    *psGradPara = 
    {
        (double)PyFloat_AsDouble(args[3]),
        (double)PyFloat_AsDouble(args[4]),
        (double)PyFloat_AsDouble(args[5]),
        FLAG_MAXG0,
        FLAG_MAXG1
    };

    return true;
}

class ExFunc: public TrajFunc
{
public:
    ExFunc
    (
        PyObject* ppyoGetK,
        PyObject* ppyoGetDkDp,
        PyObject* ppyoGetD2kDp2,
        double dP0, double dP1
    )
    {
        m_ppyoGetK = ppyoGetK;
        m_ppyoGetDkDp = ppyoGetDkDp;
        m_ppyoGetD2kDp2 = ppyoGetD2kDp2;
        m_dP0 = dP0;
        m_dP1 = dP1;
    }

    bool getK(v3* pv3K, double dP) const
    {
        PyObject* ppyoV3 = PyObject_CallOneArg(m_ppyoGetK, PyFloat_FromDouble(dP));
        ppyoV3 = PyArray_FROM_OTF(ppyoV3, NPY_FLOAT64, NPY_ARRAY_CARRAY);
        if (PyArray_SIZE((PyArrayObject*)ppyoV3) != 3)
        {
            PyErr_SetString(PyExc_RuntimeError, "the return value of getK / getDkDp / getD2kDp2 must be size-3.\n");
            PyErr_PrintEx(-1);
            std::abort();
            return false;
        }

        pv3K->m_dX = *(double*)PyArray_GETPTR1((PyArrayObject*)ppyoV3, 0);
        pv3K->m_dY = *(double*)PyArray_GETPTR1((PyArrayObject*)ppyoV3, 1);
        pv3K->m_dZ = *(double*)PyArray_GETPTR1((PyArrayObject*)ppyoV3, 2);

        return true;
    }

    bool getDkDp(v3* pv3K, double dP) const
    {
        if (m_ppyoGetDkDp == Py_None)
        {
            return TrajFunc::getDkDp(pv3K, dP);
        }

        PyObject* ppyoV3 = PyObject_CallOneArg(m_ppyoGetDkDp, PyFloat_FromDouble(dP));
        ppyoV3 = PyArray_FROM_OTF(ppyoV3, NPY_FLOAT64, NPY_ARRAY_CARRAY);
        if (PyArray_SIZE((PyArrayObject*)ppyoV3) != 3)
        {
            PyErr_SetString(PyExc_RuntimeError, "the return value of getK / getDkDp / getD2kDp2 must be size-3.\n");
            PyErr_PrintEx(-1);
            std::abort();
            return false;
        }

        pv3K->m_dX = *(double*)PyArray_GETPTR1((PyArrayObject*)ppyoV3, 0);
        pv3K->m_dY = *(double*)PyArray_GETPTR1((PyArrayObject*)ppyoV3, 1);
        pv3K->m_dZ = *(double*)PyArray_GETPTR1((PyArrayObject*)ppyoV3, 2);

        return true;
    }

    bool getD2kDp2(v3* pv3K, double dP) const
    {
        if (m_ppyoGetD2kDp2 == Py_None)
        {
            return TrajFunc::getD2kDp2(pv3K, dP);
        }
        
        PyObject* ppyoV3 = PyObject_CallOneArg(m_ppyoGetD2kDp2, PyFloat_FromDouble(dP));
        ppyoV3 = PyArray_FROM_OTF(ppyoV3, NPY_FLOAT64, NPY_ARRAY_CARRAY);
        if (PyArray_SIZE((PyArrayObject*)ppyoV3) != 3)
        {
            PyErr_SetString(PyExc_RuntimeError, "the return value of getK / getDkDp / getD2kDp2 must be size-3.\n");
            PyErr_PrintEx(-1);
            std::abort();
            return false;
        }

        pv3K->m_dX = *(double*)PyArray_GETPTR1((PyArrayObject*)ppyoV3, 0);
        pv3K->m_dY = *(double*)PyArray_GETPTR1((PyArrayObject*)ppyoV3, 1);
        pv3K->m_dZ = *(double*)PyArray_GETPTR1((PyArrayObject*)ppyoV3, 2);

        return true;
    }
protected:
    PyObject* m_ppyoGetK;
    PyObject* m_ppyoGetDkDp;
    PyObject* m_ppyoGetD2kDp2;
};

class ExTraj: public MrTraj
{
public:
    ExTraj(const GeoPara& sGeoPara, const GradPara& sGradPara, PyObject* ppyoGetK, PyObject* ppyoGetDkDp, PyObject* ppyoGetD2kDp2, double dP0, double dP1)
    {
        m_sGeoPara = sGeoPara;
        m_sGradPara = sGradPara;
        m_lNAcq = 1;
        
        const double& dSLim = m_sGradPara.dSLim;
        const double& dGLim = m_sGradPara.dGLim;
        const double& dDt = m_sGradPara.dDt;

        ptfTrajFunc = new ExFunc
        (
            ppyoGetK,
            ppyoGetDkDp,
            ppyoGetD2kDp2,
            dP0,
            dP1
        );

        GradGen gg(ptfTrajFunc, dSLim, dGLim, dDt, 8);
        gg.compute(&m_vv3Grad);
    }

    ~ExTraj()
    {
        delete ptfTrajFunc;
    }

    bool getM0PE(v3* pv3M0PE, int64_t lIAcq) const
    {
        bool bRet = true;
        bRet &= ptfTrajFunc->getK0(pv3M0PE);
        return bRet;
    }

    bool getGRO(vv3* pvv3GRO, int64_t lIAcq) const
    {
        bool bRet = true;
        *pvv3GRO = m_vv3Grad;
        return bRet;
    }

    bool getM0SP(v3* pv3M0PE, int64_t lIAcq) const
    {
        bool bRet = true;
        *pv3M0PE = v3(0,0,0);
        return bRet;
    }

    int64_t getNWaitAdc(int64_t lIAcq) const
    {
        return 0;
    }

    int64_t getNSampAdc(int64_t lIAcq) const
    {
        return m_vv3Grad.size();
    }

private:
    TrajFunc* ptfTrajFunc;
    vv3 m_vv3Grad;
};

PyObject* calGrad(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 11);

    MrTraj::GeoPara sGeoPara;
    MrTraj::GradPara sGradPara;
    getGeoGradPara(args, &sGeoPara, &sGradPara);

    double dP0 = (double)PyFloat_AsDouble(args[9]);
    double dP1 = (double)PyFloat_AsDouble(args[10]);

    ExTraj traj
    (
        sGeoPara, sGradPara,
        args[6], args[7], args[8], 
        dP0, dP1
    );
    
    vv3 vv3G;
    traj.getGRO(&vv3G, 0);

    return cvtVv3toNparr(vv3G);
}

bool getGrad_Main(MrTraj* pmt, vv3* pvv3K0, vvv3* pvvv3G, bool bShuf)
{
    bool bRet = true;
    int64_t lNAcq = pmt->getNAcq();
    double dDt = pmt->getGradPara().dDt;
    pvv3K0->resize(lNAcq);
    pvvv3G->resize(lNAcq);

    bShuf = false; // test
	vl vlShufIdx; MrTraj::genRandIdx(&vlShufIdx, lNAcq);
    for (int64_t i = 0; i < lNAcq; ++i)
    {
        int64_t _i = bShuf?vlShufIdx[i]:i;
        
        // get M0PE and GRO
        bRet &= pmt->getM0PE(&pvv3K0->at(i), _i);
        bRet &= pmt->getGRO(&pvvv3G->at(i), _i);

        // crop gradient as requested
        v3& v3K0 = pvv3K0->at(i);
        vv3& vv3G = pvvv3G->at(i);
        int64_t lNWait = pmt->getNWaitAdc(_i);
        int64_t lNSamp = pmt->getNSampAdc(_i);
        for (int64_t j = 0; j < lNWait; ++j)
        {
            v3K0 = v3K0 + (vv3G[j] + vv3G[j+1])*dDt/2e0;
        }
        vv3G = vv3(vv3G.begin()+lNWait, vv3G.begin()+lNWait+lNSamp);

        // reverse gradient if needed
        if (FLAG_REVERSE) bRet &= GradGen::revGrad(&pvv3K0->at(i), &pvvv3G->at(i), pvv3K0->at(i), pvvv3G->at(i), dDt);
    }
    return bRet;
}

PyObject* getG_Spiral(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 7);

    MrTraj::GeoPara sGeoPara;
    MrTraj::GradPara sGradPara;
    getGeoGradPara(args, &sGeoPara, &sGradPara);

    double dRhoPhi = (double)PyFloat_AsDouble(args[6]);
    Spiral traj(sGeoPara, sGradPara, dRhoPhi);
    if (FLAG_GOLDANG) traj.setRotAngInc(traj.getNRot());

    vv3 vv3K0;
    vvv3 vvv3Grad;
    getGrad_Main(&traj, &vv3K0, &vvv3Grad, !FLAG_GOLDANG);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3Grad));
}

PyObject* getG_VarDenSpiral(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 8);

    MrTraj::GeoPara sGeoPara;
    MrTraj::GradPara sGradPara;
    getGeoGradPara(args, &sGeoPara, &sGradPara);

    double dRhoPhi0 = (double)PyFloat_AsDouble(args[6]);
    double dRhoPhi1 = (double)PyFloat_AsDouble(args[7]);
    VarDenSpiral traj(sGeoPara, sGradPara, dRhoPhi0, dRhoPhi1);
    if (FLAG_GOLDANG) traj.setRotAngInc(traj.getNRot());

    vv3 vv3K0;
    vvv3 vvv3Grad;
    getGrad_Main(&traj, &vv3K0, &vvv3Grad, !FLAG_GOLDANG);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3Grad));
}

PyObject* getG_Rosette(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 9);

    MrTraj::GeoPara sGeoPara;
    MrTraj::GradPara sGradPara;
    getGeoGradPara(args, &sGeoPara, &sGradPara);

    double dOm1 = (double)PyFloat_AsDouble(args[6]);
    double dOm2 = (double)PyFloat_AsDouble(args[7]);
    double dTmax = (double)PyFloat_AsDouble(args[8]);

    Rosette traj(sGeoPara, sGradPara, dOm1, dOm2, dTmax);
    if (FLAG_GOLDANG) traj.setRotAngInc(traj.getNRot());

    vv3 vv3K0;
    vvv3 vvv3Grad;
    getGrad_Main(&traj, &vv3K0, &vvv3Grad, !FLAG_GOLDANG);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3Grad));
}

PyObject* getG_Rosette_Trad(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 9);

    MrTraj::GeoPara sGeoPara;
    MrTraj::GradPara sGradPara;
    getGeoGradPara(args, &sGeoPara, &sGradPara);

    double dOm1 = (double)PyFloat_AsDouble(args[6]);
    double dOm2 = (double)PyFloat_AsDouble(args[7]);
    double dTmax = (double)PyFloat_AsDouble(args[8]);

    Rosette_Trad traj(sGeoPara, sGradPara, dOm1, dOm2, dTmax);
    if (FLAG_GOLDANG) traj.setRotAngInc(traj.getNRot());

    vv3 vv3K0;
    vvv3 vvv3Grad;
    getGrad_Main(&traj, &vv3K0, &vvv3Grad, !FLAG_GOLDANG);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3Grad));
}

PyObject* getG_Shell3d(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 7);

    MrTraj::GeoPara sGeoPara;
    MrTraj::GradPara sGradPara;
    getGeoGradPara(args, &sGeoPara, &sGradPara);

    double dRhoTht = (double)PyFloat_AsDouble(args[6]);
    Shell3d traj(sGeoPara, sGradPara, dRhoTht);

    vv3 vv3K0;
    vvv3 vvv3Grad;
    getGrad_Main(&traj, &vv3K0, &vvv3Grad, true);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3Grad));
}

PyObject* getG_Yarnball(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 7);

    MrTraj::GeoPara sGeoPara;
    MrTraj::GradPara sGradPara;
    getGeoGradPara(args, &sGeoPara, &sGradPara);

    double dRhoPhi = (double)PyFloat_AsDouble(args[6]);
    Yarnball traj(sGeoPara, sGradPara, dRhoPhi);

    vv3 vv3K0;
    vvv3 vvv3Grad;
    getGrad_Main(&traj, &vv3K0, &vvv3Grad, true);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3Grad));
}

PyObject* getG_Seiffert(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 8);

    MrTraj::GeoPara sGeoPara;
    MrTraj::GradPara sGradPara;
    getGeoGradPara(args, &sGeoPara, &sGradPara);

    double dM = (double)PyFloat_AsDouble(args[6]);
    double dUMax = (double)PyFloat_AsDouble(args[7]);
    Seiffert traj(sGeoPara, sGradPara, dM, dUMax);

    vv3 vv3K0;
    vvv3 vvv3Grad;
    getGrad_Main(&traj, &vv3K0, &vvv3Grad, true);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3Grad));
}

PyObject* getG_Cones(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 7);

    MrTraj::GeoPara sGeoPara;
    MrTraj::GradPara sGradPara;
    getGeoGradPara(args, &sGeoPara, &sGradPara);

    double dRhoPhi = (double)PyFloat_AsDouble(args[6]);
    Cones traj(sGeoPara, sGradPara, dRhoPhi);

    vv3 vv3K0;
    vvv3 vvv3Grad;
    getGrad_Main(&traj, &vv3K0, &vvv3Grad, true);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3Grad));
}

static PyMethodDef aMeth[] = 
{
    {"calGrad", (PyCFunction)calGrad, METH_FASTCALL, ""},
    {"getG_Spiral", (PyCFunction)getG_Spiral, METH_FASTCALL, ""},
    {"getG_VarDenSpiral", (PyCFunction)getG_VarDenSpiral, METH_FASTCALL, ""},
    {"getG_Rosette", (PyCFunction)getG_Rosette, METH_FASTCALL, ""},
    {"getG_Rosette_Trad", (PyCFunction)getG_Rosette_Trad, METH_FASTCALL, ""},
    {"getG_Shell3d", (PyCFunction)getG_Shell3d, METH_FASTCALL, ""},
    {"getG_Yarnball", (PyCFunction)getG_Yarnball, METH_FASTCALL, ""},
    {"getG_Seiffert", (PyCFunction)getG_Seiffert, METH_FASTCALL, ""},
    {"getG_Cones", (PyCFunction)getG_Cones, METH_FASTCALL, ""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef sMod = 
{
    PyModuleDef_HEAD_INIT,
    "ext",   /* name of module */
    NULL,
    -1,
    aMeth
};

PyMODINIT_FUNC
PyInit_ext(void)
{
    import_array();
    return PyModule_Create(&sMod);
}