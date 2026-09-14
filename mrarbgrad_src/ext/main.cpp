#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cstdio>
#include <ctime>
#include "mag/Mag.h"
#include "traj/TrajFunc.h"
#include "traj/ScanPlan.h"
#include "traj/Spiral.h"
#include "traj/Rosette.h"
#include "traj/Yarnball.h"
#include "traj/Cones.h"

typedef std::list<vv3> lvv3;

PyObject* PyArray_FromVv3(const vv3& src)
{
    int dim0 = src.size();
    // allocate numpy array
    PyObject* ndarray;
    {
        npy_intp dims[] = {dim0, 3};
        ndarray = PyArray_ZEROS(2, dims, NPY_FLOAT64, 0);
    }

    // fill the data in
    for (i64 i = 0; i < (int)src.size(); ++i)
    {
        *(f64*)PyArray_GETPTR2((PyArrayObject*)ndarray, i, 0) = src[i].x;
        *(f64*)PyArray_GETPTR2((PyArrayObject*)ndarray, i, 1) = src[i].y;
        *(f64*)PyArray_GETPTR2((PyArrayObject*)ndarray, i, 2) = src[i].z;
    }

    return ndarray;
}

PyObject* PyArray_FromVf64(const vf64& src)
{
    int dim0 = src.size();

    // allocate numpy array
    PyObject* ndarray;
    {
        npy_intp dims[] = {dim0};
        ndarray = PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);
    }

    // fill the data in
    for (i64 i = 0; i < (int)src.size(); ++i)
    {
        *(f64*)PyArray_GETPTR1((PyArrayObject*)ndarray, i) = src[i];
    }

    return ndarray;
}

PyObject* PyList_FromVvv3(const vvv3& src)
{
    PyObject* pyList = PyList_New(0);
    for (i64 i = 0; i < (int)src.size(); ++i)
    {
        PyObject* ndarray = PyArray_FromVv3(src[i]);
        PyList_Append(pyList, ndarray);
        Py_DECREF(ndarray);
    }
    return pyList;
}

PyObject* PyArray_FromV3(const v3& src)
{
    // allocate numpy array
    PyObject* ndarray;
    {
        npy_intp dims[] = {3};
        ndarray = PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);
    }

    // fill the data in
    *(f64*)PyArray_GETPTR1((PyArrayObject*)ndarray, 0) = src.x;
    *(f64*)PyArray_GETPTR1((PyArrayObject*)ndarray, 1) = src.y;
    *(f64*)PyArray_GETPTR1((PyArrayObject*)ndarray, 2) = src.z;

    return ndarray;
}

PyObject* PyList_FromVv3(const vv3& src)
{
    PyObject* pyList = PyList_New(0);
    for (i64 i = 0; i < (int)src.size(); ++i)
    {
        PyObject* ndarray = PyArray_FromV3(src[i]);
        PyList_Append(pyList, ndarray);
        Py_DECREF(ndarray);
    }
    return pyList;
}

bool PyArray_AsVv3(PyObject* src, vv3* dst)
{
    PyArrayObject* ndarray = (PyArrayObject*)PyArray_FROM_OTF(src, NPY_FLOAT64, NPY_ARRAY_C_CONTIGUOUS);
    i64 n = PyArray_DIM(ndarray, 0);
    dst->resize(n);

    for (i64 i = 0; i < n; ++i)
    {
        f64* pdThis = (f64*)PyArray_GETPTR2(ndarray, i, 0);
        dst->at(i).x = pdThis[0];
        dst->at(i).y = pdThis[1];
        dst->at(i).z = pdThis[2];
    }

    Py_DECREF(ndarray); // what if decref another?
    return true;
}

bool PyArray_AsVf64(PyObject* src, vf64* dst)
{
    i64 n = PyArray_DIM((PyArrayObject*)src, 0);
    dst->resize(n);

    for (i64 i = 0; i < n; ++i)
    {
        dst->at(i) = *(f64*)PyArray_GETPTR1((PyArrayObject*)src, i);
    }
    return true;
}

class ExFunc: public TrajFunc
{
public:
    ExFunc
    (
        PyObject* pyGetK,
        f64 p0, f64 p1
    ): TrajFunc(p0,p1), pyGetK(pyGetK)
    {}

    bool getK(v3* k, f64 p) const
    {
        if (k==NULL) return false;
        PyObject* pyP = PyFloat_FromDouble(p);
        PyObject* pyK = PyObject_CallOneArg(pyGetK, pyP);
        Py_DECREF(pyP);
        PyObject* _pyK = pyK;
        pyK = PyArray_FROM_OTF(pyK, NPY_FLOAT64, NPY_ARRAY_CARRAY);
        Py_DECREF(_pyK);
        
        k->x = 0; k->y = 0; k->z = 0;
        i64 size = PyArray_SIZE((PyArrayObject*)pyK);
        if (size>=1) k->x = *(f64*)PyArray_GETPTR1((PyArrayObject*)pyK, 0);
        if (size>=2) k->y = *(f64*)PyArray_GETPTR1((PyArrayObject*)pyK, 1);
        if (size>=3) k->z = *(f64*)PyArray_GETPTR1((PyArrayObject*)pyK, 2);
        
        Py_DECREF(pyK);
        return true;
    }

protected:
    PyObject* pyGetK;
};

PyObject* solve_func(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    ASSERT(narg==3);
    PyObject* pyGetK = args[0];
    f64 p0 = (f64)PyFloat_AsDouble(args[1]);
    f64 p1 = (f64)PyFloat_AsDouble(args[2]);

    ExFunc func = ExFunc(pyGetK, p0, p1);
    Mag mag = Mag();
    mag.setTraj(func);
    vv3 grad(mag.lenGradRsv);
    bool retMagSolve = mag.solve(&grad, NULL);
    if (!retMagSolve) throw std::runtime_error("Mag::solve() failed.");

    return PyArray_FromVv3(grad);
}

PyObject* solve_samp(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    ASSERT(narg==1);
    PyObject* pySamp = args[0];

    vv3 samp; PyArray_AsVv3(pySamp, &samp);
    Mag mag = Mag();
    mag.setTraj(samp);
    vv3 grad(mag.lenGradRsv);
    bool retMagSolve = mag.solve(&grad, NULL);
    if (!retMagSolve) throw std::runtime_error("Mag::solve() failed.");

    return PyArray_FromVv3(grad);
}

PyObject* scan(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    const char* strTraj = PyUnicode_AsUTF8(args[0]);
    i64 nPix = PyLong_AsLong(args[1]);
    i64 nAcq = PyLong_AsLong(args[2]);

    ScanPlan* plan = NULL;
    if (strcmp(strTraj, "Spiral")==0) plan = new SpiralPlan(nPix);
    else if (strcmp(strTraj, "LVDSpiral")==0) plan = new LVDSpiralPlan(nPix);
    else if (strcmp(strTraj, "Rosette")==0) plan = new RosettePlan(nPix);
    else if (strcmp(strTraj, "RosetteClassic")==0) plan = new RosetteClassicPlan(nPix);
    else if (strcmp(strTraj, "Yarnball")==0) plan = new YarnballPlan(nPix);
    else if (strcmp(strTraj, "Cones")==0) plan = new ConesPlan(nPix);
    else { PyErr_Format(PyExc_ValueError, "unsupported trajectory name"); return NULL; }

    PyObject* pyList = PyList_New(0);
    for (i64 iAcq=0; iAcq<nAcq; ++iAcq)
    {
        v3 k0(0), k1(0); vv3 grad(0);
        plan->getGrad(&k0, &grad, &k1, iAcq);

        PyObject* pyK0 = PyArray_FromV3(k0);
        PyObject* pyGrad = PyArray_FromVv3(grad);
        PyObject* pyK1 = PyArray_FromV3(k1);

        PyObject* pyTuple = PyTuple_Pack(3, pyK0, pyGrad, pyK1);

        Py_DECREF(pyK0);
        Py_DECREF(pyGrad);
        Py_DECREF(pyK1);

        PyList_Append(pyList, pyTuple);
        Py_DECREF(pyTuple);
    }

    delete plan;
    return pyList;
}

PyObject* config(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    ASSERT(narg==10);
    Mag::dt = PyFloat_AsDouble(args[0]);
    Mag::ovsp = PyLong_AsLong(args[1]);
    Mag::sLim = PyFloat_AsDouble(args[2]);
    Mag::gLim = PyFloat_AsDouble(args[3]);
    Mag::g0Norm = PyFloat_AsDouble(args[4]);
    Mag::g1Norm = PyFloat_AsDouble(args[5]);
    Mag::enTrajRep = args[6]==Py_True;
    Mag::enGradRep = args[7]==Py_True;
    Mag::lenGradRsv = PyLong_AsLongLong(args[8]);
    Mag::lenTrajRsv = PyLong_AsLongLong(args[9]);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* saveF64(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    ASSERT(narg==3);
    const char* strHdr = PyUnicode_AsUTF8(args[0]);
    const char* strBin = PyUnicode_AsUTF8(args[1]);
    FILE* fHdr = fopen(strHdr, "w");
    FILE* fBin = fopen(strBin, "wb");
    if (fHdr==NULL || fBin==NULL)
    {
        if (fHdr) fclose(fHdr);
        if (fBin) fclose(fBin);
        PyErr_Format
        (
             PyExc_FileNotFoundError, 
             "hdr file open %s; bin file open %s",
             fHdr!=NULL?"SUCCESS":"FAILED", 
             fBin!=NULL?"SUCCESS":"FAILED"
        );
        return NULL;
    }

    vv3 vv3Data;
    i64 n = PyList_GET_SIZE(args[2]);
    for (i64 i=0; i<n; ++i)
    {
        PyArray_AsVv3(PyList_GET_ITEM(args[2], i), &vv3Data);
        v3::saveF64(fHdr, fBin, vv3Data);
    }

    fclose(fHdr); fclose(fBin);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* loadF64(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    ASSERT(narg==2);
    const char* strHdr = PyUnicode_AsUTF8(args[0]);
    const char* strBin = PyUnicode_AsUTF8(args[1]);
    FILE* fHdr = fopen(strHdr, "r");
    FILE* fBin = fopen(strBin, "rb");
    if (fHdr==NULL || fBin==NULL)
    {
        if (fHdr) fclose(fHdr);
        if (fBin) fclose(fBin);
        PyErr_Format
        (
             PyExc_FileNotFoundError, 
             "hdr file open %s; bin file open %s",
             fHdr!=NULL?"SUCCESS":"FAILED", 
             fBin!=NULL?"SUCCESS":"FAILED"
        );
        return NULL;
    }

    lvv3 lvv3Data; vv3 vv3Data;
    while (1)
    {
        v3::loadF64(fHdr, fBin, &vv3Data);
        lvv3Data.push_back(vv3Data);
    }

    fclose(fHdr); fclose(fBin);
    vvv3 vvv3Data(lvv3Data.begin(), lvv3Data.end());
    return PyList_FromVvv3(vvv3Data);
}

PyObject* saveF32(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    ASSERT(narg==3);
    const char* strHdr = PyUnicode_AsUTF8(args[0]);
    const char* strBin = PyUnicode_AsUTF8(args[1]);
    FILE* fHdr = fopen(strHdr, "w");
    FILE* fBin = fopen(strBin, "wb");
    if (fHdr==NULL || fBin==NULL)
    {
        if (fHdr) fclose(fHdr);
        if (fBin) fclose(fBin);
        PyErr_Format
        (
             PyExc_FileNotFoundError, 
             "hdr file open %s; bin file open %s",
             fHdr!=NULL?"SUCCESS":"FAILED", 
             fBin!=NULL?"SUCCESS":"FAILED"
        );
        return NULL;
    }

    vv3 vv3Data;
    i64 n = PyList_GET_SIZE(args[2]);
    bool ret = true;
    for (i64 i=0; i<n; ++i)
    {
        PyArray_AsVv3(PyList_GET_ITEM(args[2], i), &vv3Data);
        v3::saveF32(fHdr, fBin, vv3Data);
    }

    fclose(fHdr); fclose(fBin);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject* loadF32(PyObject* self, PyObject* const* args, Py_ssize_t narg)
{
    ASSERT(narg==2);
    const char* strHdr = PyUnicode_AsUTF8(args[0]);
    const char* strBin = PyUnicode_AsUTF8(args[1]);
    typedef std::list<vv3> lvv3;
    FILE* fHdr = fopen(strHdr, "r");
    FILE* fBin = fopen(strBin, "rb");
    if (fHdr==NULL || fBin==NULL)
    {
        if (fHdr) fclose(fHdr);
        if (fBin) fclose(fBin);
        PyErr_Format
        (
             PyExc_FileNotFoundError, 
             "hdr file open %s; bin file open %s",
             fHdr!=NULL?"SUCCESS":"FAILED", 
             fBin!=NULL?"SUCCESS":"FAILED"
        );
        return NULL;
    }

    lvv3 lvv3Data; vv3 vv3Data;
    while (1)
    {
        v3::loadF32(fHdr, fBin, &vv3Data);
        lvv3Data.push_back(vv3Data);
    }

    fclose(fHdr); fclose(fBin);
    vvv3 vvv3Data(lvv3Data.begin(), lvv3Data.end());
    return PyList_FromVvv3(vvv3Data);
}

static PyMethodDef methods[] = 
{
    {"solve_func", (PyCFunction)solve_func, METH_FASTCALL, ""},
    {"solve_samp", (PyCFunction)solve_samp, METH_FASTCALL, ""},
    {"scan", (PyCFunction)scan, METH_FASTCALL, ""},
    {"config", (PyCFunction)config, METH_FASTCALL, ""},
    {"saveF64", (PyCFunction)saveF64, METH_FASTCALL, ""},
    {"loadF64", (PyCFunction)loadF64, METH_FASTCALL, ""},
    {"saveF32", (PyCFunction)saveF32, METH_FASTCALL, ""},
    {"loadF32", (PyCFunction)loadF32, METH_FASTCALL, ""},
    {NULL, NULL, 0, NULL}        /* Tree Sentinel */
};

static struct PyModuleDef module = 
{
    PyModuleDef_HEAD_INIT,
    "ext",   /* name of module */
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC
PyInit_ext(void)
{
    import_array();
    return PyModule_Create(&module);
}
