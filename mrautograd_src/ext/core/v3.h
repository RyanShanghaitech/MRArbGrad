#pragma once

#include <cmath>
#include <vector>
#include <array>
#include <cstdio>
#include <cstdint>
#include "global.h"

class v3
{
public:
    typedef std::vector<v3> vv3;
    typedef std::vector<vv3> vvv3;

    double m_dX;
    double m_dY;
    double m_dZ;

    v3();
    v3(double dX, double dY, double dZ);
    ~v3();
    v3 operator+(const v3 &rhs) const;
    v3 operator+(const double &rhs) const;
    v3 operator-(const v3 &rhs) const;
    v3 operator-(const double &rhs) const;
    v3 operator*(const v3 &rhs) const;
    v3 operator*(const double &rhs) const;
    v3 operator/(const v3 &rhs) const;
    v3 operator/(const double &rhs) const;
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
        std::vector<v3>* pvv3Dst,
        int iAx, double dAng,
        const std::vector<v3>& vv3Src
    );
    static bool saveF64(FILE* pfBHdr, FILE* pfBin, vv3& vv3Data);
    static bool saveF32(FILE* pfBHdr, FILE* pfBin, vv3& vv3Data);
    static bool saveI16(FILE* pfBHdr, FILE* pfBin, vv3& vv3Data);
private:
    static bool genRotMat(std::array<v3,3>* pav3RotMat, int iAx, double dAng);
};