#pragma once

#include "Intp.h"

class LinIntp : public Intp
{
public:
    LinIntp() {}

    LinIntp(const vf64& vf64X, const vf64& vf64Y)
    {
        fit(vf64X, vf64Y);
    }

    virtual bool fit(const vf64& vf64X, const vf64& vf64Y)
    {
        m_vf64X = vf64X;
        m_vf64Y = vf64Y;

        if (!validate(m_vf64X, m_vf64Y))
        {
            m_vf64X.clear();
            m_vf64Y.clear();
            throw std::invalid_argument("!validate(m_vf64X, m_vf64Y)");
        }

        m_idxCache = 0;

        const i64 num = (i64)m_vf64X.size();
        m_vf64Slope.resize(num - 1);

        for (i64 i = 0; i < num - 1; ++i)
        {
            const f64 dx = m_vf64X[i + 1] - m_vf64X[i];
            m_vf64Slope[i] = (m_vf64Y[i + 1] - m_vf64Y[i]) / dx;
        }

        return true;
    }

    virtual f64 eval(f64 x, i64 ord = 0) const
    {
        if (m_vf64X.size() < 2) throw std::runtime_error("m_vf64X.size()");

        const i64 idx = getIdx(x);
        const f64 dx = x - m_vf64X[idx];

        if (ord == 0)
        {
            return m_vf64Y[idx] + m_vf64Slope[idx] * dx;
        }
        if (ord == 1)
        {
            return m_vf64Slope[idx];
        }

        return 0e0;
    }

private:
    vf64 m_vf64Slope;
};
