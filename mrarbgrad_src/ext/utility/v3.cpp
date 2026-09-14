#include "v3.h"
#include <cstring> // test

v3::v3() :x(0e0), y(0e0), z(0e0) {}
v3::v3(f64 xyz) :x(xyz), y(xyz), z(xyz) {}
v3::v3(f64 x, f64 y, f64 z) :x(x), y(y), z(z) {}
v3::~v3() {}

v3 v3::operator+(const v3 &rhs) const
{
    return v3
    (
        this->x + rhs.x,
        this->y + rhs.y,
        this->z + rhs.z
    );
}

v3& v3::operator+=(const v3 &rhs)
{
    this->x += rhs.x;
    this->y += rhs.y;
    this->z += rhs.z;
    return *this;
}

v3 v3::operator+(const f64 &rhs) const
{
    return v3
    (
        this->x + rhs,
        this->y + rhs,
        this->z + rhs
    );
}

v3& v3::operator+=(const f64 &rhs)
{
    this->x += rhs;
    this->y += rhs;
    this->z += rhs;
    return *this;
}

v3 v3::operator-(const v3 &rhs) const
{
    return v3
    (
        this->x - rhs.x,
        this->y - rhs.y,
        this->z - rhs.z
    );
}

v3& v3::operator-=(const v3 &rhs)
{
    this->x -= rhs.x;
    this->y -= rhs.y;
    this->z -= rhs.z;
    return *this;
}

v3 v3::operator-(const f64 &rhs) const
{
    return v3
    (
        this->x - rhs,
        this->y - rhs,
        this->z - rhs
    );
}

v3& v3::operator-=(const f64 &rhs)
{
    this->x -= rhs;
    this->y -= rhs;
    this->z -= rhs;
    return *this;
}

v3 v3::operator*(const v3 &rhs) const
{
    return v3
    (
        this->x * rhs.x,
        this->y * rhs.y,
        this->z * rhs.z
    );
}

v3& v3::operator*=(const v3 &rhs)
{
    this->x *= rhs.x;
    this->y *= rhs.y;
    this->z *= rhs.z;
    return *this;
}

v3 v3::operator*(const f64 &rhs) const
{
    return v3
    (
        this->x * rhs,
        this->y * rhs,
        this->z * rhs
    );
}

v3& v3::operator*=(const f64 &rhs)
{
    this->x *= rhs;
    this->y *= rhs;
    this->z *= rhs;
    return *this;
}

v3 v3::operator/(const v3 &rhs) const
{
    return v3
    (
        this->x / rhs.x,
        this->y / rhs.y,
        this->z / rhs.z
    );
}

v3& v3::operator/=(const v3 &rhs)
{
    this->x /= rhs.x;
    this->y /= rhs.y;
    this->z /= rhs.z;
    return *this;
}

v3 v3::operator/(const f64 &rhs) const
{
    return v3
    (
        this->x / rhs,
        this->y / rhs,
        this->z / rhs
    );
}

v3& v3::operator/=(const f64 &rhs)
{
    this->x /= rhs;
    this->y /= rhs;
    this->z /= rhs;
    return *this;
}

bool v3::operator==(const v3 &rhs) const
{
    return bool
    (
        this->x == rhs.x &&
        this->y == rhs.y &&
        this->z == rhs.z
    );
}

bool v3::operator!=(const v3 &rhs) const
{
    return bool
    (
        this->x != rhs.x ||
        this->y != rhs.y ||
        this->z != rhs.z
    );
}

f64& v3::operator[](i64 idx)
{
    if (idx==0 || idx==-3) return x;
    if (idx==1 || idx==-2) return y;
    if (idx==2 || idx==-1) return z;
    throw std::runtime_error("idx");
}

f64 v3::operator[](i64 idx) const
{
    if (idx==0 || idx==-3) return x;
    if (idx==1 || idx==-2) return y;
    if (idx==2 || idx==-1) return z;
    throw std::runtime_error("idx");
}

f64 v3::norm(const v3& in)
{
    return sqrt
    (
        in.x*in.x +
        in.y*in.y +
        in.z*in.z
    );
}

v3 v3::cross(const v3& in0, const v3& in1)
{
    return v3
    (
        in0.y*in1.z - in0.z*in1.y,
        -in0.x*in1.z + in0.z*in1.x,
        in0.x*in1.y - in0.y*in1.x
    );
}

f64 v3::inner(const v3& in0, const v3& in1)
{
    return f64
    (
        in0.x*in1.x +
        in0.y*in1.y +
        in0.z*in1.z
    );
}

v3 v3::pow(const v3& in, f64 exp)
{
    return v3
    (
        std::pow(in.x, exp),
        std::pow(in.y, exp),
        std::pow(in.z, exp)
    );
}

bool v3::rotate
(
    v3* dst,
    int ax, f64 ang,
    const v3& src
)
{
    if (!dst) return false;
    bool ret = true;

    std::array<v3,3> av3RotMat;
    ret &= genRotMat(&av3RotMat, ax, ang);
    if (!ret) return ret;

    *dst = v3
    (
        v3::inner(av3RotMat[0], src),
        v3::inner(av3RotMat[1], src),
        v3::inner(av3RotMat[2], src)
    );

    return ret;
}

bool v3::rotate
(
    vv3* dst,
    int ax, f64 ang,
    const vv3& src
)
{
    if (!dst) return false;
    bool ret = true;

    std::array<v3, 3> av3RotMat;
    ret &= genRotMat(&av3RotMat, ax, ang);
    if (!ret) return ret;

    if (dst->size() != src.size()) {
        dst->resize(src.size());
    }

    for (size_t i = 0; i < src.size(); ++i)
    {
        f64 tx = v3::inner(av3RotMat[0], src[i]);
        f64 ty = v3::inner(av3RotMat[1], src[i]);
        f64 tz = v3::inner(av3RotMat[2], src[i]);

        (*dst)[i].x = tx;
        (*dst)[i].y = ty;
        (*dst)[i].z = tz;
    }

    return true;
}

bool v3::rotate
(
    lv3* dst,
    int ax, f64 ang,
    const lv3& src
)
{
    if (!dst) return false;
    bool ret = true;

    std::array<v3,3> av3RotMat;
    ret &= genRotMat(&av3RotMat, ax, ang);
    if (!ret) return ret;

    // apply rotation matrix
    lv3 _lv3Dst; // for self-in self-out compatible
    lv3::const_iterator ilv3CoordSrc = src.begin();
    while (ilv3CoordSrc != src.end())
    {
        _lv3Dst.push_back
        (
            v3
            (
                v3::inner(av3RotMat[0], *ilv3CoordSrc),
                v3::inner(av3RotMat[1], *ilv3CoordSrc),
                v3::inner(av3RotMat[2], *ilv3CoordSrc)
            )
        );

        ++ilv3CoordSrc;
    }
    dst->swap(_lv3Dst);

    return ret;
}

bool v3::linspace(vv3* dst, const v3& start, const v3& stop, i64 n, bool end)
{
    bool ret = true;
    dst->resize(n);
    v3 diff = stop - start;
    f64 deno = end ? std::max(n-1,(i64)1) : n;
    for (i64 i = 0; i < n; ++i)
    { (*dst)[i] = start + diff * i/deno; }
    return ret;
}

vv3 v3::linspace(const v3& start, const v3& stop, i64 n, bool end)
{
    vv3 ret(n);
    v3::linspace(&ret, start, stop, n, end);
    return ret;
}

v3 v3::axisroll(const v3& in, i64 shift)
{
    v3 out;
    switch ((shift%3+3)%3)
    {
    case 1:
        out.x = in.y;
        out.y = in.z;
        out.z = in.x;
        break;
        
    case 2:
        out.x = in.z;
        out.y = in.x;
        out.z = in.y;
        break;
    
    default:
        out = in;
        break;
    }
    return out;
}

bool v3::saveF64(FILE* hdr, FILE* bin, const vv3& data)
{
    bool ret = true;
    i64 lenData = data.size();
    fprintf(hdr, "float64[%ld][3];\n", (long)lenData);

    f64* bufFile = (f64*)malloc(lenData*3*sizeof(f64));
    for(i64 i=0; i<(i64)lenData; ++i)
    {
        for(i64 j=0; j<3; ++j)
        { bufFile[3*i+j] = (f64)data[i][j]; }
    }
    ret &= (i64)fwrite(bufFile, sizeof(f64), lenData*3, bin)==lenData*3;
    free(bufFile);
    return ret;
}

bool v3::loadF64(FILE* hdr, FILE* bin, vv3* data)
{
    bool ret = true;
    data->clear();
    i64 lenData = 0;
    {
        long _;
        int nRead = fscanf(hdr, "float64[%ld][3];\n", &_);
        if (nRead == EOF) return true; // EOF
        else if (nRead != 1) return false;
        lenData = (i64)_;
    }
    data->resize(lenData);

    f64* bufFile = (f64*)malloc(lenData*3*sizeof(f64));
    ret &= (i64)fread(bufFile, sizeof(f64), lenData*3, bin)==lenData*3;
    for(i64 i=0; i<lenData; ++i)
    {
        for(i64 j=0; j<3; ++j)
        { (*data)[i][j] = (f64)bufFile[3*i+j]; }
    }
    free(bufFile);
    return ret;
}

bool v3::saveF32(FILE* hdr, FILE* bin, const vv3& data)
{
    bool ret = true;
    i64 lenData = data.size();
    fprintf(hdr, "float32[%ld][3];\n", (long)lenData);

    f32* bufFile = (f32*)malloc(lenData*3*sizeof(f32));
    for(i64 i=0; i<(i64)lenData; ++i)
    {
        for(i64 j=0; j<3; ++j)
        { bufFile[3*i+j] = (f32)data[i][j]; }
    }
    ret &= (i64)fwrite(bufFile, sizeof(f32), lenData*3, bin)==lenData*3;
    free(bufFile);
    return ret;
}

bool v3::loadF32(FILE* hdr, FILE* bin, vv3* data)
{
    bool ret = true;
    data->clear();
    i64 lenData = 0;
    {
        long _;
        int nRead = fscanf(hdr, "float32[%ld][3];\n", &_);
        if (nRead == EOF) return true; // EOF
        else if (nRead != 1) return false;
        lenData = (i64)_;
    }
    data->resize(lenData);

    f32* bufFile = (f32*)malloc(lenData*3*sizeof(f32));
    ret &= (i64)fread(bufFile, sizeof(f32), lenData*3, bin)==lenData*3;
    for(i64 i=0; i<lenData; ++i)
    {
        for(i64 j=0; j<3; ++j)
        { (*data)[i][j] = (f64)bufFile[3*i+j]; }
    }
    free(bufFile);
    return ret;
}

bool v3::genRotMat(std::array<v3,3>* rotMag, int ax, f64 ang)
{
    if (!rotMag) return false;
    switch (ax)
    {
    case 0:
        (*rotMag)[0] = v3(1e0, 0e0, 0e0);
        (*rotMag)[1] = v3(0e0, std::cos(ang), -std::sin(ang));
        (*rotMag)[2] = v3(0e0, std::sin(ang), std::cos(ang));
        break;
    case 1:
        (*rotMag)[0] = v3(std::cos(ang), 0e0, std::sin(ang));
        (*rotMag)[1] = v3(0e0, 1e0, 0e0);
        (*rotMag)[2] = v3(-std::sin(ang), 0e0, std::cos(ang));
        break;
    case 2:
        (*rotMag)[0] = v3(std::cos(ang), -std::sin(ang), 0e0);
        (*rotMag)[1] = v3(std::sin(ang), std::cos(ang), 0e0);
        (*rotMag)[2] = v3(0e0, 0e0, 1e0);
        break;
    default:
        return false;
    }

    return true;
}

