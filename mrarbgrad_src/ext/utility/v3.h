#pragma once

#include <cmath>
#include <list>
#include <array>
#include "global.h"

class v3;

typedef std::vector<v3> vv3;
typedef std::list<v3> lv3;

class v3
{
public:
    f64 x, y, z;

    v3();
    v3(f64 xyz);
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
    static f64 norm(const v3& in);
    static v3 cross(const v3& in0, const v3& in1);
    static f64 inner(const v3& in0, const v3& in1);
    static v3 pow(const v3& in, f64 exp);
    static bool rotate
    (
        v3* dst,
        int ax, f64 ang,
        const v3& src
    );
    static bool rotate
    (
        vv3* dst,
        int ax, f64 ang,
        const vv3& src
    );
    static bool rotate
    (
        lv3* dst,
        int ax, f64 ang,
        const lv3& src
    );
    static bool linspace(vv3* dst, const v3& start, const v3& stop, i64 n, bool end=false);
    static vv3 linspace(const v3& start, const v3& stop, i64 n, bool end=false);
    static v3 axisroll(const v3& in, i64 shift);
    static bool saveF64(FILE* hdr, FILE* bin, const vv3& data);
    static bool loadF64(FILE* hdr, FILE* bin, vv3* data);
    static bool saveF32(FILE* hdr, FILE* bin, const vv3& data);
    static bool loadF32(FILE* hdr, FILE* bin, vv3* data);
private:
    static bool genRotMat(std::array<v3,3>* rotMat, int ax, f64 ang);
};
