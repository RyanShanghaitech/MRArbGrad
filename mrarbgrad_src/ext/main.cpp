#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstdio>
#include <ctime>
#include <algorithm>
#include <iostream>
#include "mag/Mag.h"
#include "traj/TrajFunc.h"
#include "traj/MrTraj.h"
#include "traj/Spiral.h"
#include "traj/VDSpiral.h"
#include "traj/Rosette.h"
#include "traj/Shell3d.h"
#include "traj/Yarnball.h"
#include "traj/Seiffert.h"
#include "traj/Cones.h"
#include "utility/SplineIntp.h"

bool gMain_enTrajRev (0);
bool gMain_enGoldAng (0);
bool gMain_enShuffle (0);

PyObject* cvtVv3toNpa(vv3& vv3Src)
{
    int dim0 = vv3Src.size();
    // allocate numpy array
    PyObject* pNumpyArray;
    {
        npy_intp dims[] = {dim0, 3};
        pNumpyArray = PyArray_ZEROS(2, dims, NPY_FLOAT64, 0);
    }

    // fill the data in
    for (i64 i = 0; i < (int)vv3Src.size(); ++i)
    {
        *(f64*)PyArray_GETPTR2((PyArrayObject*)pNumpyArray, i, 0) = vv3Src[i].x;
        *(f64*)PyArray_GETPTR2((PyArrayObject*)pNumpyArray, i, 1) = vv3Src[i].y;
        *(f64*)PyArray_GETPTR2((PyArrayObject*)pNumpyArray, i, 2) = vv3Src[i].z;
    }

    return pNumpyArray;
}

PyObject* cvtVf64toNpa(const std::vector<f64>& vf64Src)
{
    int dim0 = vf64Src.size();

    // allocate numpy array
    PyObject* pNumpyArray;
    {
        npy_intp dims[] = {dim0};
        pNumpyArray = PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);
    }

    // fill the data in
    for (i64 i = 0; i < (int)vf64Src.size(); ++i)
    {
        *(f64*)PyArray_GETPTR1((PyArrayObject*)pNumpyArray, i) = vf64Src[i];
    }

    return pNumpyArray;
}

PyObject* cvtVvv3toList(vvv3& vvv3Src)
{
    PyObject* pPyList = PyList_New(0);
    for (i64 i = 0; i < (int)vvv3Src.size(); ++i)
    {
        PyObject* pNumpyArray = cvtVv3toNpa(vvv3Src[i]);
        PyList_Append(pPyList, pNumpyArray);
        Py_DECREF(pNumpyArray);
    }
    return pPyList;
}

PyObject* cvtV3toNpa(v3& v3Src)
{
    // allocate numpy array
    PyObject* pNumpyArray;
    {
        npy_intp dims[] = {3};
        pNumpyArray = PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);
    }

    // fill the data in
    *(f64*)PyArray_GETPTR1((PyArrayObject*)pNumpyArray, 0) = v3Src.x;
    *(f64*)PyArray_GETPTR1((PyArrayObject*)pNumpyArray, 1) = v3Src.y;
    *(f64*)PyArray_GETPTR1((PyArrayObject*)pNumpyArray, 2) = v3Src.z;

    return pNumpyArray;
}

PyObject* cvtVv3toList(vv3& vv3Src)
{
    PyObject* pPyList = PyList_New(0);
    for (i64 i = 0; i < (int)vv3Src.size(); ++i)
    {
        PyObject* pNumpyArray = cvtV3toNpa(vv3Src[i]);
        PyList_Append(pPyList, pNumpyArray);
        Py_DECREF(pNumpyArray);
    }
    return pPyList;
}

bool cvtNpa2Vv3(PyObject* pNumpyArray, vv3* pvv3Out)
{
    PyArrayObject* ppyaoNpa = (PyArrayObject*)PyArray_FROM_OTF(pNumpyArray, NPY_FLOAT64, NPY_ARRAY_C_CONTIGUOUS);
    i64 n = PyArray_DIM(ppyaoNpa, 0);
    pvv3Out->resize(n);

    for (i64 i = 0; i < n; ++i)
    {
        f64* pdThis = (f64*)PyArray_GETPTR2(ppyaoNpa, i, 0);
        pvv3Out->at(i).x = pdThis[0];
        pvv3Out->at(i).y = pdThis[1];
        pvv3Out->at(i).z = pdThis[2];
    }

    Py_DECREF(ppyaoNpa); // what if decref another?
    return true;
}

bool cvtNpa2Vf64(PyObject* pNumpyArray, vf64* pvf64Out)
{
    i64 n = PyArray_DIM((PyArrayObject*)pNumpyArray, 0);
    pvf64Out->resize(n);

    for (i64 i = 0; i < n; ++i)
    {
        pvf64Out->at(i) = *(f64*)PyArray_GETPTR1((PyArrayObject*)pNumpyArray, i);
    }
    return true;
}

bool inline checkNarg(i64 nArg, i64 nArgExp)
{
    if (nArg != nArgExp)
    {
        printf("wrong num. of arg, narg=%ld, %ld expected\n", nArg, nArgExp);
        abort();
        return false;
    }
    return true;
}

bool getGeoGradPara(PyObject* const* args, MrTraj::GeoPara* pobjGeoPara, MrTraj::GradPara* pobjGradPara)
{
    *pobjGeoPara = 
    {
        (f64)PyFloat_AsDouble(args[0]),
        (i64)PyLong_AsLong(args[1])
    };

    *pobjGradPara = 
    {
        (f64)PyFloat_AsDouble(args[2]),
        (f64)PyFloat_AsDouble(args[3]),
        (f64)PyFloat_AsDouble(args[4])
    };

    return true;
}

class ExFunc: public TrajFunc
{
public:
    ExFunc
    (
        PyObject* pPyObj_getK,
        PyObject* pPyObj_getDkDp,
        PyObject* pPyObj_getD2kDp2,
        f64 p0, f64 p1
    ):
        TrajFunc(p0,p1)
    {
        m_pPyObj_getK = pPyObj_getK;
        m_pPyObj_getDkDp = pPyObj_getDkDp;
        m_pPyObj_getD2kDp2 = pPyObj_getD2kDp2;
    }
    
    bool getK(v3* k, f64 p)
    {
        PyObject* pPyObj_p = PyFloat_FromDouble(p);
        PyObject* pPyObj_v3 = PyObject_CallOneArg(m_pPyObj_getK, pPyObj_p);
        Py_DECREF(pPyObj_p);
        PyObject* _pPyObj_v3 = pPyObj_v3;
        pPyObj_v3 = PyArray_FROM_OTF(pPyObj_v3, NPY_FLOAT64, NPY_ARRAY_CARRAY);
        Py_DECREF(_pPyObj_v3);
        if (PyArray_SIZE((PyArrayObject*)pPyObj_v3) != 3)
        {
            PyErr_SetString(PyExc_RuntimeError, "the return value of getK / getDkDp / getD2kDp2 must be size-3.\n");
            PyErr_PrintEx(-1);
            std::abort();
            return false;
        }

        k->x = *(f64*)PyArray_GETPTR1((PyArrayObject*)pPyObj_v3, 0);
        k->y = *(f64*)PyArray_GETPTR1((PyArrayObject*)pPyObj_v3, 1);
        k->z = *(f64*)PyArray_GETPTR1((PyArrayObject*)pPyObj_v3, 2);

        return true;
    }

    bool getDkDp(v3* k, f64 p)
    {
        if (m_pPyObj_getDkDp == Py_None)
        {
            return TrajFunc::getDkDp(k, p);
        }

        PyObject* pPyObj_p = PyFloat_FromDouble(p);
        PyObject* pPyObj_v3 = PyObject_CallOneArg(m_pPyObj_getDkDp, pPyObj_p);
        Py_DECREF(pPyObj_p);
        PyObject* _pPyObj_v3 = pPyObj_v3;
        pPyObj_v3 = PyArray_FROM_OTF(pPyObj_v3, NPY_FLOAT64, NPY_ARRAY_CARRAY);
        Py_DECREF(_pPyObj_v3);
        if (PyArray_SIZE((PyArrayObject*)pPyObj_v3) != 3)
        {
            PyErr_SetString(PyExc_RuntimeError, "the return value of getK / getDkDp / getD2kDp2 must be size-3.\n");
            PyErr_PrintEx(-1);
            std::abort();
            return false;
        }

        k->x = *(f64*)PyArray_GETPTR1((PyArrayObject*)pPyObj_v3, 0);
        k->y = *(f64*)PyArray_GETPTR1((PyArrayObject*)pPyObj_v3, 1);
        k->z = *(f64*)PyArray_GETPTR1((PyArrayObject*)pPyObj_v3, 2);

        return true;
    }

    bool getD2kDp2(v3* k, f64 p)
    {
        if (m_pPyObj_getD2kDp2 == Py_None)
        {
            return TrajFunc::getD2kDp2(k, p);
        }
        
        PyObject* pPyObj_p = PyFloat_FromDouble(p);
        PyObject* pPyObj_v3 = PyObject_CallOneArg(m_pPyObj_getD2kDp2, pPyObj_p);
        Py_DECREF(pPyObj_p);
        PyObject* _pPyObj_v3 = pPyObj_v3;
        pPyObj_v3 = PyArray_FROM_OTF(pPyObj_v3, NPY_FLOAT64, NPY_ARRAY_CARRAY);
        Py_DECREF(_pPyObj_v3);
        if (PyArray_SIZE((PyArrayObject*)pPyObj_v3) != 3)
        {
            PyErr_SetString(PyExc_RuntimeError, "the return value of getK / getDkDp / getD2kDp2 must be size-3.\n");
            PyErr_PrintEx(-1);
            std::abort();
            return false;
        }

        k->x = *(f64*)PyArray_GETPTR1((PyArrayObject*)pPyObj_v3, 0);
        k->y = *(f64*)PyArray_GETPTR1((PyArrayObject*)pPyObj_v3, 1);
        k->z = *(f64*)PyArray_GETPTR1((PyArrayObject*)pPyObj_v3, 2);

        return true;
    }
protected:
    PyObject* m_pPyObj_getK;
    PyObject* m_pPyObj_getDkDp;
    PyObject* m_pPyObj_getD2kDp2;
};

class ExTraj: public MrTraj
{
public:
    ExTraj(const GeoPara& objGeoPara, const GradPara& objGradPara, PyObject* pPyObj_getK, PyObject* pPyObj_getDkDp, PyObject* pPyObj_getD2kDp2, f64 p0, f64 p1):
        MrTraj(objGeoPara,objGradPara,1,0),
        ptfTrajFunc(NULL)
    {   
        ptfTrajFunc = new ExFunc
        (
            pPyObj_getK,
            pPyObj_getDkDp,
            pPyObj_getD2kDp2,
            p0,
            p1
        );

        TIC;
        calGRO(&m_vv3G, &m_vf64P, *ptfTrajFunc, m_objGradPara, 8);
        TOC;
        m_nSampMax = m_vv3G.size();
    }

    ExTraj(const GeoPara& objGeoPara, const GradPara& objGradPara, vv3& vv3K):
        MrTraj(objGeoPara,objGradPara,1,0),
        ptfTrajFunc(NULL)
    {
        TIC;
        calGRO(&m_vv3G, &m_vf64P, vv3K, m_objGradPara, 8);
        TOC;
        m_nSampMax = m_vv3G.size();
    }

    ~ExTraj()
    {
        if (ptfTrajFunc)
        {
            delete ptfTrajFunc;
            ptfTrajFunc = NULL;
        }
    }

    virtual bool getGRO(vv3* pvv3G, i64 iAcq)
    {
        *pvv3G = m_vv3G;
        return true;
    }

    virtual bool getM0PE(v3* pv3M0PE, i64 iAcq)
    {
        ptfTrajFunc->getK0(pv3M0PE);
        return true;
    }

    bool getPRO(vf64* vf64P, i64 iAcq) // get parameter sequence of GRO
    {
        *vf64P = m_vf64P;
        return true;
    }

private:
    TrajFunc* ptfTrajFunc;
    vv3 m_vv3G;
    vf64 m_vf64P;
};

PyObject* calGrad4ExFunc(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 10);

    MrTraj::GeoPara objGeoPara;
    MrTraj::GradPara objGradPara;
    getGeoGradPara(args, &objGeoPara, &objGradPara);

    f64 p0 = (f64)PyFloat_AsDouble(args[8]);
    f64 p1 = (f64)PyFloat_AsDouble(args[9]);

    ExTraj traj
    (
        objGeoPara, objGradPara,
        args[5], args[6], args[7], 
        p0, p1
    );

    vv3 vv3G;
    traj.getGRO(&vv3G, 0);
    vf64 vf64P;
    traj.getPRO(&vf64P, 0);

    return Py_BuildValue("OO", cvtVv3toNpa(vv3G), cvtVf64toNpa(vf64P));
}

PyObject* calGrad4ExSamp(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    try
    {
        checkNarg(narg, 6);

        MrTraj::GeoPara objGeoPara;
        MrTraj::GradPara objGradPara;
        getGeoGradPara(args, &objGeoPara, &objGradPara);

        vv3 vv3K; cvtNpa2Vv3(args[5], &vv3K);

        ExTraj traj
        (
            objGeoPara, objGradPara,
            vv3K
        );
        
        vv3 vv3G;
        traj.getGRO(&vv3G, 0);
        vf64 vf64P;
        traj.getPRO(&vf64P, 0);

        return Py_BuildValue("OO", cvtVv3toNpa(vv3G), cvtVf64toNpa(vf64P));
    }
    catch (const std::exception& e)
    {
        return PyErr_Format(PyExc_RuntimeError, "Exception: %s", e.what());
    }
}

bool getG(MrTraj* pmt, vv3* pvv3M0PE, vvv3* pvvv3GRO)
{
    bool ret = true;
    i64 nAcq = pmt->getNAcq();
    f64 dt = pmt->getGradPara().dt;
    pvv3M0PE->resize(nAcq);
    pvvv3GRO->resize(nAcq);

    bool& enShuf = gMain_enShuffle;
	vi64 vi64ShufSeq; MrTraj::genRandIdx(&vi64ShufSeq, nAcq);
    for (i64 i = 0; i < nAcq; ++i)
    {
        i64 _i = enShuf?vi64ShufSeq[i]:i;
        
        // get M0PE and GRO
        vv3 vv3GRO; ret &= pmt->getGRO(&vv3GRO, _i);
        v3 v3M0PE; ret &= pmt->getM0PE(&v3M0PE, _i);

        // reverse gradient if needed
        if (gMain_enTrajRev) ret &= Mag::revGrad(&v3M0PE, &vv3GRO, v3M0PE, vv3GRO, dt);

        pvv3M0PE->at(i) = v3M0PE;
        pvvv3GRO->at(i) = vv3GRO;
    }
    return ret;
}

PyObject* getG_Spiral(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 7);
    
    MrTraj::GeoPara objGeoPara;
    MrTraj::GradPara objGradPara;
    getGeoGradPara(args, &objGeoPara, &objGradPara);

    i64 nSlice = (i64)PyLong_AsLong(args[5]);
    f64 kRhoPhi = (f64)PyFloat_AsDouble(args[6]);
    Spiral traj(objGeoPara, objGradPara, nSlice, kRhoPhi);
    if (gMain_enGoldAng) traj.setRotAng(GOLDANG);

    vv3 vv3K0;
    vvv3 vvv3G;
    getG(&traj, &vv3K0, &vvv3G);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3G));
}

PyObject* getG_VDSpiral(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 8);

    MrTraj::GeoPara objGeoPara;
    MrTraj::GradPara objGradPara;
    getGeoGradPara(args, &objGeoPara, &objGradPara);

    i64 nSlice = (i64)PyLong_AsLong(args[5]);
    f64 kRhoPhi0 = (f64)PyFloat_AsDouble(args[6]);
    f64 kRhoPhi1 = (f64)PyFloat_AsDouble(args[7]);
    VDSpiral traj(objGeoPara, objGradPara, nSlice, kRhoPhi0, kRhoPhi1);
    if (gMain_enGoldAng) traj.setRotAng(GOLDANG);

    vv3 vv3K0;
    vvv3 vvv3G;
    getG(&traj, &vv3K0, &vvv3G);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3G));
}

PyObject* getG_VDSpiral_RT(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 8);

    MrTraj::GeoPara objGeoPara;
    MrTraj::GradPara objGradPara;
    getGeoGradPara(args, &objGeoPara, &objGradPara);

    f64 kRhoPhi0 = (f64)PyFloat_AsDouble(args[5]);
    f64 kRhoPhi1 = (f64)PyFloat_AsDouble(args[6]);
    i64 nAcq = (i64)PyLong_AsLong(args[7]);
    VDSpiral_RT traj(objGeoPara, objGradPara, kRhoPhi0, kRhoPhi1, nAcq);

    vv3 vv3K0;
    vvv3 vvv3G;
    getG(&traj, &vv3K0, &vvv3G);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3G));
}

PyObject* getG_Rosette(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 9);

    MrTraj::GeoPara objGeoPara;
    MrTraj::GradPara objGradPara;
    getGeoGradPara(args, &objGeoPara, &objGradPara);

    i64 nSlice = (i64)PyLong_AsLong(args[5]);
    f64 om1 = (f64)PyFloat_AsDouble(args[6]);
    f64 om2 = (f64)PyFloat_AsDouble(args[7]);
    f64 tMax = (f64)PyFloat_AsDouble(args[8]);

    Rosette traj(objGeoPara, objGradPara, nSlice, om1, om2, tMax);
    // printf("Rosette DTE: %e s\n", traj.getAvrDTE());
    if (gMain_enGoldAng) traj.setRotAng(GOLDANG);

    vv3 vv3K0;
    vvv3 vvv3G;
    getG(&traj, &vv3K0, &vvv3G);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3G));
}

PyObject* getG_Rosette_Trad(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 10);

    MrTraj::GeoPara objGeoPara;
    MrTraj::GradPara objGradPara;
    getGeoGradPara(args, &objGeoPara, &objGradPara);

    i64 nSlice = (i64)PyLong_AsLong(args[5]);
    f64 om1 = (f64)PyFloat_AsDouble(args[6]);
    f64 om2 = (f64)PyFloat_AsDouble(args[7]);
    f64 tMax = (f64)PyFloat_AsDouble(args[8]);
    f64 dTE = (f64)PyFloat_AsDouble(args[9]);

    Rosette_Trad traj(objGeoPara, objGradPara, nSlice, om1, om2, tMax, dTE);
    if (gMain_enGoldAng) traj.setRotAng(GOLDANG);

    vv3 vv3K0;
    vvv3 vvv3G;
    getG(&traj, &vv3K0, &vvv3G);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3G));
}

PyObject* getG_Shell3d(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 6);

    MrTraj::GeoPara objGeoPara;
    MrTraj::GradPara objGradPara;
    getGeoGradPara(args, &objGeoPara, &objGradPara);

    f64 kRhoTht = (f64)PyFloat_AsDouble(args[5]);
    Shell3d traj(objGeoPara, objGradPara, kRhoTht);

    vv3 vv3K0;
    vvv3 vvv3G;
    getG(&traj, &vv3K0, &vvv3G);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3G));
}

PyObject* getG_Yarnball(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 6);

    MrTraj::GeoPara objGeoPara;
    MrTraj::GradPara objGradPara;
    getGeoGradPara(args, &objGeoPara, &objGradPara);

    f64 kRhoPhi = (f64)PyFloat_AsDouble(args[5]);
    Yarnball traj(objGeoPara, objGradPara, kRhoPhi);

    vv3 vv3K0;
    vvv3 vvv3G;
    getG(&traj, &vv3K0, &vvv3G);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3G));
}

PyObject* getG_Seiffert(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 7);

    MrTraj::GeoPara objGeoPara;
    MrTraj::GradPara objGradPara;
    getGeoGradPara(args, &objGeoPara, &objGradPara);

    f64 m = (f64)PyFloat_AsDouble(args[5]);
    f64 uMax = (f64)PyFloat_AsDouble(args[6]);
    Seiffert traj(objGeoPara, objGradPara, m, uMax);

    vv3 vv3K0;
    vvv3 vvv3G;
    getG(&traj, &vv3K0, &vvv3G);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3G));
}

PyObject* getG_Cones(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 6);

    MrTraj::GeoPara objGeoPara;
    MrTraj::GradPara objGradPara;
    getGeoGradPara(args, &objGeoPara, &objGradPara);

    f64 kRhoPhi = (f64)PyFloat_AsDouble(args[5]);
    Cones traj(objGeoPara, objGradPara, kRhoPhi);

    vv3 vv3K0;
    vvv3 vvv3G;
    getG(&traj, &vv3K0, &vvv3G);

    return Py_BuildValue("OO", cvtVv3toList(vv3K0), cvtVvv3toList(vvv3G));
}

PyObject* setSolverMtg(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    extern bool gMrTraj_enMtg;
    checkNarg(narg, 1);
    gMrTraj_enMtg = PyLong_AsLong(args[0]);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* setTrajRev(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 1);
    gMain_enTrajRev = PyLong_AsLong(args[0]);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* setGoldAng(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 1);
    gMain_enGoldAng = PyLong_AsLong(args[0]);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* setShuf(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 1);
    gMain_enShuffle = PyLong_AsLong(args[0]);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* setMaxG0(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    extern f64 gMrTraj_g0Norm;
    checkNarg(narg, 1);
    bool enMaxG0 = PyLong_AsLong(args[0]);
    if (enMaxG0) gMrTraj_g0Norm = 1e6;
    else gMrTraj_g0Norm = 0e0;
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* setMaxG1(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    extern f64 gMrTraj_g1Norm;
    checkNarg(narg, 1);
    bool enMaxG1 = PyLong_AsLong(args[0]);
    if (enMaxG1) gMrTraj_g1Norm = 1e6;
    else gMrTraj_g1Norm = 0e0;
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* setMagOverSamp(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    extern i64 gMag_oversamp;
    checkNarg(narg, 1);
    gMag_oversamp = PyLong_AsLong(args[0]);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* setMagSFS(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    extern bool gMag_enSFS;
    checkNarg(narg, 1);
    gMag_enSFS = PyLong_AsLong(args[0]);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* setMagGradRep(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    extern bool gMag_enGradRep;
    checkNarg(narg, 1);
    gMag_enGradRep = PyLong_AsLong(args[0]);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* setMagTrajRep(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    extern bool gMag_enTrajRep;
    checkNarg(narg, 1);
    gMag_enTrajRep = PyLong_AsLong(args[0]);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* setDbgPrint(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    extern bool glob_enDbgPrint;
    checkNarg(narg, 1);
    glob_enDbgPrint = PyLong_AsLong(args[0]);
    Py_INCREF(Py_None);
    return Py_None;
}

vv3 vv3Test;
PyObject* getTestVal(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    checkNarg(narg, 0);
    return Py_BuildValue("O", cvtVv3toNpa(vv3Test));
}

static PyMethodDef aMeth[] = 
{
    {"calGrad4ExFunc", (PyCFunction)calGrad4ExFunc, METH_FASTCALL, ""},
    {"calGrad4ExSamp", (PyCFunction)calGrad4ExSamp, METH_FASTCALL, ""},
    {"getG_Spiral", (PyCFunction)getG_Spiral, METH_FASTCALL, ""},
    {"getG_VDSpiral", (PyCFunction)getG_VDSpiral, METH_FASTCALL, ""},
    {"getG_VDSpiral_RT", (PyCFunction)getG_VDSpiral_RT, METH_FASTCALL, ""},
    {"getG_Rosette", (PyCFunction)getG_Rosette, METH_FASTCALL, ""},
    {"getG_Rosette_Trad", (PyCFunction)getG_Rosette_Trad, METH_FASTCALL, ""},
    {"getG_Shell3d", (PyCFunction)getG_Shell3d, METH_FASTCALL, ""},
    {"getG_Yarnball", (PyCFunction)getG_Yarnball, METH_FASTCALL, ""},
    {"getG_Seiffert", (PyCFunction)getG_Seiffert, METH_FASTCALL, ""},
    {"getG_Cones", (PyCFunction)getG_Cones, METH_FASTCALL, ""},
    {"setSolverMtg", (PyCFunction)setSolverMtg, METH_FASTCALL, ""},
    {"setTrajRev", (PyCFunction)setTrajRev, METH_FASTCALL, ""},
    {"setGoldAng", (PyCFunction)setGoldAng, METH_FASTCALL, ""},
    {"setShuf", (PyCFunction)setShuf, METH_FASTCALL, ""},
    {"setMaxG0", (PyCFunction)setMaxG0, METH_FASTCALL, ""},
    {"setMaxG1", (PyCFunction)setMaxG1, METH_FASTCALL, ""},
    {"getTestVal", (PyCFunction)getTestVal, METH_FASTCALL, ""},
    {"setMagOverSamp", (PyCFunction)setMagOverSamp, METH_FASTCALL, ""},
    {"setMagSFS", (PyCFunction)setMagSFS, METH_FASTCALL, ""},
    {"setMagGradRep", (PyCFunction)setMagGradRep, METH_FASTCALL, ""},
    {"setMagTrajRep", (PyCFunction)setMagTrajRep, METH_FASTCALL, ""},
    {"setDbgPrint", (PyCFunction)setDbgPrint, METH_FASTCALL, ""},
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