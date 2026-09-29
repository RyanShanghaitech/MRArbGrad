#pragma once

#include <stdint.h>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "../utility/global.h"

class Intp
{
public:
    enum SearchMode
    {
        EBinary = 0,
        ECached,
        EUniform
    } eSearchMode;

    Intp() : eSearchMode(EBinary), idxCache(0) {}
    virtual ~Intp() {}

    virtual bool fit(const vf64& xs, const vf64& ys) = 0;

    virtual f64 eval(f64 x, i64 ord = 0) const = 0;
    // order: order of derivation, default is 0 (function value)

    static bool validate(const vf64& xs, const vf64& ys)
    {
        if (xs.size()!=ys.size() || xs.size()==0)
        { return false; }

        i64 nSamp = i64(xs.size());
        for (i64 i = 1; i < nSamp; ++i)
        { if (xs[i] <= xs[i-1]) return false; }
        return true;
    }

    f64 operator()(f64 x, i64 ord=0) const
    { return eval(x, ord); }

protected:
    vf64 xs, ys;
    mutable i64 idxCache;

    i64 floor(const f64& x) const
    {
        const i64 nSamp = i64(xs.size());
        ASSERT(nSamp >= 2);

        i64 idx;

        if (eSearchMode == EBinary)
        {
            i64 low = 0;
            i64 high = nSamp - 1;
            while (high - low > 1)
            {
                i64 mid = (low + high) / 2;
                if (xs[mid] > x) high = mid;
                else low = mid;
            }
            idx = low;
            return idx;
        }
        if (eSearchMode == ECached)
        {
            if (idxCache < 0) idxCache = 0;
            if (idxCache > nSamp - 2) idxCache = nSamp - 2;
            idx = idxCache;
            while (idx > 0 && xs[idx] > x) --idx;
            while (idx < nSamp - 2 && xs[idx +1] < x) ++idx;
            idxCache = idx;
            return idx;
        }
        if (eSearchMode == EUniform)
        {
            const f64 x0 = xs.front();
            const f64 x1 = xs.back();
            idx = i64((x - x0) / (x1 - x0) * (nSamp - 1));
            if (idx < 0) idx = 0;
            if (idx > nSamp - 2) idx = nSamp - 2;
            return idx;
        }

        throw std::invalid_argument("m_eSearchMode");
    }
};
