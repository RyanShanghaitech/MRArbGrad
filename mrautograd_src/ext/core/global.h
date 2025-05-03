#pragma once

#if (__cplusplus >= 201103L)

#include <cmath>
template<typename T>
inline T round(T x)
{
    return std::round(x);
}

#else

template<typename T>
inline T round(T x)
{
    return (x >= 0) ? std::floor(x + T(0.5)) : std::ceil(x - T(0.5));
}

#endif

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#define PRINT(X) printf("%s: %.3e\n", #X, (double)(X));
