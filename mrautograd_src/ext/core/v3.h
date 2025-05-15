#pragma once

#include <cmath>
#include <vector>
#include <list>
#include <array>
#include <cstdio>
#include <cstdint>
#include "global.h"

class v3
{
public:
    typedef std::vector<v3> vv3;
    typedef std::list<v3> lv3;

    double m_dX;
    double m_dY;
    double m_dZ;

    v3();
    v3(double dX, double dY, double dZ);
    ~v3();
    v3 operator+(const v3 &rhs) const;
    v3 operator+=(const v3 &rhs);
    v3 operator+(const double &rhs) const;
    v3 operator+=(const double &rhs);
    v3 operator-(const v3 &rhs) const;
    v3 operator-=(const v3 &rhs);
    v3 operator-(const double &rhs) const;
    v3 operator-=(const double &rhs);
    v3 operator*(const v3 &rhs) const;
    v3 operator*=(const v3 &rhs);
    v3 operator*(const double &rhs) const;
    v3 operator*=(const double &rhs);
    v3 operator/(const v3 &rhs) const;
    v3 operator/=(const v3 &rhs);
    v3 operator/(const double &rhs) const;
    v3 operator/=(const double &rhs);
    bool operator==(const v3 &rhs) const;
    bool operator!=(const v3 &rhs) const;
    static double norm(const v3& v3_tObj);
    static v3 cross(const v3& v3_tObj0, const v3& v3_tObj1);
    static double inner(const v3& v3_tObj0, const v3& v3_tObj1);
    static v3 pow(const v3& v3_tObj, double dPow);
    static bool rotate
    (
        v3* pv3Dst,
        int iAx, double dAng,
        const v3& v3Src
    );
    static bool rotate
    (
        vv3* pvv3Dst,
        int iAx, double dAng,
        const vv3& vv3Src
    );
    static bool rotate
    (
        lv3* plv3Dst,
        int iAx, double dAng,
        const lv3& lv3Src
    );
    template<typename cv3>
    static bool saveF64(FILE* pfBHdr, FILE* pfBin, const cv3& cv3Data);
    template<typename cv3>
    static bool saveF32(FILE* pfBHdr, FILE* pfBin, const cv3& cv3Data);
    template<typename cv3>
    static bool saveI16(FILE* pfBHdr, FILE* pfBin, const cv3& cv3Data);
private:
    static bool genRotMat(std::array<v3,3>* pav3RotMat, int iAx, double dAng);
};

template<typename cv3>
bool v3::saveF64(FILE* pfBHdr, FILE* pfBin, const cv3& cv3Data)
{
    bool bRet = true;
    fprintf(pfBHdr, "float64[%ld][3];\n", (int64_t)cv3Data.size());
    typename cv3::const_iterator icv3Data = cv3Data.begin();
    while (icv3Data!=cv3Data.end())
    {
        bRet &= (fwrite(&icv3Data->m_dX, sizeof(double), 1, pfBin) == 1);
        bRet &= (fwrite(&icv3Data->m_dY, sizeof(double), 1, pfBin) == 1);
        bRet &= (fwrite(&icv3Data->m_dZ, sizeof(double), 1, pfBin) == 1);
        ++icv3Data;
    }
    return bRet;
}

template<typename cv3>
bool v3::saveF32(FILE* pfBHdr, FILE* pfBin, const cv3& cv3Data)
{
    bool bRet = true;
    fprintf(pfBHdr, "float32[%ld][3];\n", (int64_t)cv3Data.size());
    typename cv3::const_iterator icv3Data = cv3Data.begin();
    float f32X, f32Y, f32Z;
    while (icv3Data!=cv3Data.end())
    {
        f32X = float(icv3Data->m_dX);
        f32Y = float(icv3Data->m_dY);
        f32Z = float(icv3Data->m_dZ);
        bRet &= (fwrite(&f32X, sizeof(float), 1, pfBin) == 1);
        bRet &= (fwrite(&f32Y, sizeof(float), 1, pfBin) == 1);
        bRet &= (fwrite(&f32Z, sizeof(float), 1, pfBin) == 1);
        ++icv3Data;
    }
    return bRet;
}

template<typename cv3>
bool v3::saveI16(FILE* pfBHdr, FILE* pfBin, const cv3& cv3Data)
{
    bool bRet = true;
    typename cv3::const_iterator icv3Data = cv3Data.begin();
    double dRef = 0e0;
    while (icv3Data!=cv3Data.end())
    {
        dRef = std::max(dRef, v3::norm(*icv3Data));
        ++icv3Data;
    }
    fprintf(pfBHdr, "int16[%ld][3]; // Ref:%.6e\n", (int64_t)cv3Data.size(), dRef);
    icv3Data = cv3Data.begin();
    int16_t i16X, i16Y, i16Z;
    while (icv3Data!=cv3Data.end())
    {
        i16X = (int16_t)round(icv3Data->m_dX/dRef * 0x7fff);
        i16Y = (int16_t)round(icv3Data->m_dY/dRef * 0x7fff);
        i16Z = (int16_t)round(icv3Data->m_dZ/dRef * 0x7fff);
        bRet &= (fwrite(&i16X, sizeof(int16_t), 1, pfBin) == 1);
        bRet &= (fwrite(&i16Y, sizeof(int16_t), 1, pfBin) == 1);
        bRet &= (fwrite(&i16Z, sizeof(int16_t), 1, pfBin) == 1);
        ++icv3Data;
    }
    return bRet;
}