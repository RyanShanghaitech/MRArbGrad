#include "v3.h"
#include <array>

v3::v3() :m_dX(0e0), m_dY(0e0), m_dZ(0e0) {}
v3::v3(double dX, double dY, double dZ) :m_dX(dX), m_dY(dY), m_dZ(dZ) {}
v3::~v3() {}

v3 v3::operator+(const v3 &rhs) const
{
    return v3
    (
        this->m_dX + rhs.m_dX,
        this->m_dY + rhs.m_dY,
        this->m_dZ + rhs.m_dZ
    );
}

v3 v3::operator+(const double &rhs) const
{
    return v3
    (
        this->m_dX + rhs,
        this->m_dY + rhs,
        this->m_dZ + rhs
    );
}

v3 v3::operator-(const v3 &rhs) const
{
    return v3
    (
        this->m_dX - rhs.m_dX,
        this->m_dY - rhs.m_dY,
        this->m_dZ - rhs.m_dZ
    );
}

v3 v3::operator-(const double &rhs) const
{
    return v3
    (
        this->m_dX - rhs,
        this->m_dY - rhs,
        this->m_dZ - rhs
    );
}

v3 v3::operator*(const v3 &rhs) const
{
    return v3
    (
        this->m_dX * rhs.m_dX,
        this->m_dY * rhs.m_dY,
        this->m_dZ * rhs.m_dZ
    );
}

v3 v3::operator*(const double &rhs) const
{
    return v3
    (
        this->m_dX * rhs,
        this->m_dY * rhs,
        this->m_dZ * rhs
    );
}

v3 v3::operator/(const v3 &rhs) const
{
    return v3
    (
        this->m_dX / rhs.m_dX,
        this->m_dY / rhs.m_dY,
        this->m_dZ / rhs.m_dZ
    );
}

v3 v3::operator/(const double &rhs) const
{
    return v3
    (
        this->m_dX / rhs,
        this->m_dY / rhs,
        this->m_dZ / rhs
    );
}

bool v3::operator==(const v3 &rhs) const
{
    return bool
    (
        this->m_dX == rhs.m_dX &&
        this->m_dY == rhs.m_dY &&
        this->m_dZ == rhs.m_dZ
    );
}

bool v3::operator!=(const v3 &rhs) const
{
    return bool
    (
        this->m_dX != rhs.m_dX ||
        this->m_dY != rhs.m_dY ||
        this->m_dZ != rhs.m_dZ
    );
}

double v3::norm(const v3& v3_tObj)
{
    return sqrt
    (
        v3_tObj.m_dX*v3_tObj.m_dX +
        v3_tObj.m_dY*v3_tObj.m_dY +
        v3_tObj.m_dZ*v3_tObj.m_dZ
    );
}

v3 v3::cross(const v3& v3_tObj0, const v3& v3_tObj1)
{
    return v3
    (
        v3_tObj0.m_dY*v3_tObj1.m_dZ - v3_tObj0.m_dZ*v3_tObj1.m_dY,
        -v3_tObj0.m_dX*v3_tObj1.m_dZ + v3_tObj0.m_dZ*v3_tObj1.m_dX,
        v3_tObj0.m_dX*v3_tObj1.m_dY - v3_tObj0.m_dY*v3_tObj1.m_dX
    );
}

double v3::inner(const v3& v3_tObj0, const v3& v3_tObj1)
{
    return double
    (
        v3_tObj0.m_dX*v3_tObj1.m_dX +
        v3_tObj0.m_dY*v3_tObj1.m_dY +
        v3_tObj0.m_dZ*v3_tObj1.m_dZ
    );
}

v3 v3::pow(const v3& v3_tObj, double dPow)
{
    return v3
    (
        std::pow(v3_tObj.m_dX, dPow),
        std::pow(v3_tObj.m_dY, dPow),
        std::pow(v3_tObj.m_dZ, dPow)
    );
}

bool v3::genRotMat(std::array<v3,3>* pav3RotMat, int iAx, double dAng)
{
    switch (iAx)
    {
    case 0:
        (*pav3RotMat)[0] = v3(1e0, 0e0, 0e0);
        (*pav3RotMat)[1] = v3(0e0, std::cos(dAng), -std::sin(dAng));
        (*pav3RotMat)[2] = v3(0e0, std::sin(dAng), std::cos(dAng));
        break;
    case 1:
        (*pav3RotMat)[0] = v3(std::cos(dAng), 0e0, std::sin(dAng));
        (*pav3RotMat)[1] = v3(0e0, 1e0, 0e0);
        (*pav3RotMat)[2] = v3(-std::sin(dAng), 0e0, std::cos(dAng));
        break;
    case 2:
        (*pav3RotMat)[0] = v3(std::cos(dAng), -std::sin(dAng), 0e0);
        (*pav3RotMat)[1] = v3(std::sin(dAng), std::cos(dAng), 0e0);
        (*pav3RotMat)[2] = v3(0e0, 0e0, 1e0);
        break;
    default:
        return false;
    }

    return true;
}

bool v3::rotate
(
    v3* pv3Dst,
    int iAx, double dAng,
    const v3& v3Src
)
{
    bool bRet = true;

    std::array<v3,3> av3RotMat;
    bRet &= genRotMat(&av3RotMat, iAx, dAng);

    *pv3Dst = v3
    (
        v3::inner(av3RotMat[0], v3Src),
        v3::inner(av3RotMat[1], v3Src),
        v3::inner(av3RotMat[2], v3Src)
    );

    return bRet;
}

bool v3::rotate
(
    std::vector<v3>* pvv3Dst,
    int iAx, double dAng,
    const std::vector<v3>& vv3Src
)
{
    bool bRet = true;

    std::array<v3,3> av3RotMat;
    bRet &= genRotMat(&av3RotMat, iAx, dAng);

    // apply rotation matrix
    std::vector<v3> vv3Dst(vv3Src.size());
    std::vector<v3>::const_iterator ivv3CoordSrc = vv3Src.begin();
    std::vector<v3>::iterator ivv3CoordDst = vv3Dst.begin();
    while (ivv3CoordDst != vv3Dst.end())
    {
        *ivv3CoordDst = v3
        (
            v3::inner(av3RotMat[0], *ivv3CoordSrc),
            v3::inner(av3RotMat[1], *ivv3CoordSrc),
            v3::inner(av3RotMat[2], *ivv3CoordSrc)
        );
        ++ivv3CoordSrc;
        ++ivv3CoordDst;
    }
    *pvv3Dst = vv3Dst;

    return bRet;
}

bool v3::saveF64(FILE* pfBHdr, FILE* pfBin, vv3& vv3Data)
{
    bool bRet = true;
    fprintf(pfBHdr, "float64[%ld][3];\n", (int64_t)vv3Data.size());
    for (int64_t i = 0; i < (int64_t)vv3Data.size(); ++i)
    {
        bRet &= (fwrite(&vv3Data[i].m_dX, sizeof(double), 1, pfBin) == 1);
        bRet &= (fwrite(&vv3Data[i].m_dY, sizeof(double), 1, pfBin) == 1);
        bRet &= (fwrite(&vv3Data[i].m_dZ, sizeof(double), 1, pfBin) == 1);
    }
    return bRet;
}

bool v3::saveF32(FILE* pfBHdr, FILE* pfBin, vv3& vv3Data)
{
    bool bRet = true;
    fprintf(pfBHdr, "float32[%ld][3];\n", (int64_t)vv3Data.size());
    
    float f32X, f32Y, f32Z;
    for (int64_t i = 0; i < (int64_t)vv3Data.size(); ++i)
    {
        f32X = float(vv3Data[i].m_dX);
        f32Y = float(vv3Data[i].m_dY);
        f32Z = float(vv3Data[i].m_dZ);
        bRet &= (fwrite(&f32X, sizeof(float), 1, pfBin) == 1);
        bRet &= (fwrite(&f32Y, sizeof(float), 1, pfBin) == 1);
        bRet &= (fwrite(&f32Z, sizeof(float), 1, pfBin) == 1);
    }
    return bRet;
}

bool v3::saveI16(FILE* pfBHdr, FILE* pfBin, vv3& vv3Data)
{
    bool bRet = true;
    
    double dRef = 0e0;
    for (int64_t i = 0; i < (int64_t)vv3Data.size(); ++i)
    {
        dRef = std::max(dRef, v3::norm(vv3Data[i]));
    }
    fprintf(pfBHdr, "int16[%ld][3]; // Ref:%.6e\n", (int64_t)vv3Data.size(), dRef);

    int16_t i16X, i16Y, i16Z;
    for (int64_t i = 0; i < (int64_t)vv3Data.size(); ++i)
    {
        i16X = (int16_t)round(vv3Data[i].m_dX/dRef * 0x7fff);
        i16Y = (int16_t)round(vv3Data[i].m_dY/dRef * 0x7fff);
        i16Z = (int16_t)round(vv3Data[i].m_dZ/dRef * 0x7fff);
        bRet &= (fwrite(&i16X, sizeof(int16_t), 1, pfBin) == 1);
        bRet &= (fwrite(&i16Y, sizeof(int16_t), 1, pfBin) == 1);
        bRet &= (fwrite(&i16Z, sizeof(int16_t), 1, pfBin) == 1);
    }
    return bRet;
}