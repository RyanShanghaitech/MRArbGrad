#pragma once

#include <cmath>
#include <list>
#include <array>
#include "global.h"

class v3;

typedef std::vector<v3> vv3;
typedef std::vector<vv3> vvv3;
typedef std::list<v3> lv3;

class v3
{
public:
    f64 x, y, z;

    v3();
    v3(f64 _);
    v3(f64 x, f64 y, f64 z);
    ~v3();
    v3 operator+(const v3 &rhs) const;
    v3& operator+=(const v3 &rhs);
    v3 operator+(const f64 &rhs) const;
    v3& operator+=(const f64 &rhs);
    v3 operator-(const v3 &rhs) const;
    v3& operator-=(const v3 &rhs);
    v3 operator-(const f64 &rhs) const;
    v3& operator-=(const f64 &rhs);
    v3 operator*(const v3 &rhs) const;
    v3& operator*=(const v3 &rhs);
    v3 operator*(const f64 &rhs) const;
    v3& operator*=(const f64 &rhs);
    v3 operator/(const v3 &rhs) const;
    v3& operator/=(const v3 &rhs);
    v3 operator/(const f64 &rhs) const;
    v3& operator/=(const f64 &rhs);
    bool operator==(const v3 &rhs) const;
    bool operator!=(const v3 &rhs) const;
    f64& operator[](i64 idx);
    f64 operator[](i64 idx) const;
    static f64 norm(const v3& v3In);
    static v3 cross(const v3& v3In0, const v3& v3In1);
    static f64 inner(const v3& v3In0, const v3& v3In1);
    static v3 pow(const v3& v3In, f64 exp);
    static bool rotate
    (
        v3* pv3Dst,
        int iAx, f64 ang,
        const v3& v3Src
    );
    static bool rotate
    (
        vv3* pvv3Dst,
        int iAx, f64 ang,
        const vv3& vv3Src
    );
    static bool rotate
    (
        lv3* plv3Dst,
        int iAx, f64 ang,
        const lv3& lv3Src
    );
    static v3 axisroll(const v3& v3In, i64 lShift);
    template<typename cv3>
    static bool saveF64(FILE* pfBHdr, FILE* pfBin, const cv3& cv3Data);
    template<typename cv3>
    static bool loadF64(FILE* pfBHdr, FILE* pfBin, cv3* pcv3Data);
    template<typename cv3>
    static bool saveF32(FILE* pfBHdr, FILE* pfBin, const cv3& cv3Data);
    template<typename cv3>
    static bool loadF32(FILE* pfBHdr, FILE* pfBin, cv3* pcv3Data);
private:
    static bool genRotMat(std::array<v3,3>* pav3RotMat, int iAx, f64 ang);
};

template<typename cv3>
bool v3::saveF64(FILE* pfBHdr, FILE* pfBin, const cv3& cv3Data)
{
    bool ret = true;
    fprintf(pfBHdr, "float64[%ld][3];\n", (i64)cv3Data.size());
    typename cv3::const_iterator icv3Data = cv3Data.begin();
    while (icv3Data!=cv3Data.end())
    {
        ret &= (fwrite(&icv3Data->x, sizeof(f64), 1, pfBin) == 1);
        ret &= (fwrite(&icv3Data->y, sizeof(f64), 1, pfBin) == 1);
        ret &= (fwrite(&icv3Data->z, sizeof(f64), 1, pfBin) == 1);
        ++icv3Data;
    }
    return ret;
}

template<typename cv3>
bool v3::loadF64(FILE* pfBHdr, FILE* pfBin, cv3* pcv3Data)
{
    bool ret = true;
    i64 lenData = 0;
    int nByte = fscanf(pfBHdr, "float64[%ld][3];\n", &lenData);
    if (nByte!=1) ret = false;
    pcv3Data->resize(lenData);
    typename cv3::iterator icv3Data = pcv3Data->begin();
    while (icv3Data!=pcv3Data->end())
    {
        ret &= (fread(&icv3Data->x, sizeof(f64), 1, pfBin) == 1);
        ret &= (fread(&icv3Data->y, sizeof(f64), 1, pfBin) == 1);
        ret &= (fread(&icv3Data->z, sizeof(f64), 1, pfBin) == 1);
        ++icv3Data;
    }
    return ret;
}

template<typename cv3>
bool v3::saveF32(FILE* pfBHdr, FILE* pfBin, const cv3& cv3Data)
{
    bool ret = true;
    fprintf(pfBHdr, "float32[%ld][3];\n", (i64)cv3Data.size());
    typename cv3::const_iterator icv3Data = cv3Data.begin();
    float f32X, f32Y, f32Z;
    while (icv3Data!=cv3Data.end())
    {
        f32X = (float)icv3Data->x;
        f32Y = (float)icv3Data->y;
        f32Z = (float)icv3Data->z;
        ret &= (fwrite(&f32X, sizeof(float), 1, pfBin) == 1);
        ret &= (fwrite(&f32Y, sizeof(float), 1, pfBin) == 1);
        ret &= (fwrite(&f32Z, sizeof(float), 1, pfBin) == 1);
        ++icv3Data;
    }
    return ret;
}

template<typename cv3>
bool v3::loadF32(FILE* pfBHdr, FILE* pfBin, cv3* pcv3Data)
{
    bool ret = true;
    i64 lenData = 0;
    int nByte = fscanf(pfBHdr, "float32[%ld][3];\n", &lenData);
    if (nByte!=1) ret = false;
    pcv3Data->resize(lenData);
    typename cv3::iterator icv3Data = pcv3Data->begin();
    float f32X, f32Y, f32Z;
    while (icv3Data!=pcv3Data->end())
    {
        ret &= (fread(&f32X, sizeof(float), 1, pfBin) == 1);
        ret &= (fread(&f32Y, sizeof(float), 1, pfBin) == 1);
        ret &= (fread(&f32Z, sizeof(float), 1, pfBin) == 1);

        icv3Data->x = (f64)f32X;
        icv3Data->y = (f64)f32Y;
        icv3Data->z = (f64)f32Z;

        ++icv3Data;
    }
    return ret;
}
