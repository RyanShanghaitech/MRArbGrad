#pragma once

template<typename T>
inline T round(T x)
{
    return (x >= 0) ? std::floor(x + T(0.5)) : std::ceil(x - T(0.5));
}

#undef M_PI
#define M_PI (3.14159265358979323846)

#define PRINT(X) printf("%s: %.3e\n", #X, (double)(X));
