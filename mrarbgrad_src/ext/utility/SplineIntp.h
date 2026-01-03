#pragma once

#include "Intp.h"

class SplineIntp : public Intp
{
public:
    SplineIntp() {}

    explicit SplineIntp(const vf64& vf64X, const vf64& vf64Y)
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

        const i64 num = i64(m_vf64X.size());

        vf64 vf64H(num - 1);
        for (i64 i = 0; i < num - 1; ++i)
            vf64H[i] = m_vf64X[i + 1] - m_vf64X[i];

        // Step 1: Set up the tridiagonal system
        vf64 vf64Alpha(num, 0.0);
        for (i64 i = 1; i < num - 1; ++i)
            vf64Alpha[i] = (3e0 / vf64H[i]) * (m_vf64Y[i + 1] - m_vf64Y[i]) - (3e0 / vf64H[i - 1]) * (m_vf64Y[i] - m_vf64Y[i - 1]);

        // Step 2: Solve tridiagonal system for c (second derivatives)
        vf64 vf64L(num, 1.0), vf64Mu(num, 0.0), vf64Z(num, 0.0);
        m_vf64C.resize(num, 0.0);
        m_vf64B.resize(num - 1, 0.0);
        m_vf64D.resize(num - 1, 0.0);
        m_vf64A = m_vf64Y;

        for (i64 i = 1; i < num - 1; ++i)
        {
            vf64L[i] = 2e0 * (m_vf64X[i + 1] - m_vf64X[i - 1]) - vf64H[i - 1] * vf64Mu[i - 1];
            vf64Mu[i] = vf64H[i] / vf64L[i];
            vf64Z[i] = (vf64Alpha[i] - vf64H[i - 1] * vf64Z[i - 1]) / vf64L[i];
        }

        // Natural spline boundary conditions
        vf64L[num - 1] = 1.0;
        vf64Z[num - 1] = 0.0;
        m_vf64C[num - 1] = 0.0;

        // Back substitution
        for (i64 i = num - 2; i >= 0; --i)
        {
            m_vf64C[i] = vf64Z[i] - vf64Mu[i] * m_vf64C[i + 1];
            m_vf64B[i] = (m_vf64A[i + 1] - m_vf64A[i]) / vf64H[i] - vf64H[i] * (m_vf64C[i + 1] + 2e0 * m_vf64C[i]) / 3e0;
            m_vf64D[i] = (m_vf64C[i + 1] - m_vf64C[i]) / (3e0 * vf64H[i]);
        }

        return true;
    }

    virtual f64 eval(f64 x, i64 ord = 0) const // order: order of derivation, default is 0 (function value)
    {
        if (m_vf64X.size() < 2) throw std::runtime_error("m_vdX.size()");

        i64 idx = getIdx(x);

        f64 dx = x - m_vf64X[idx];
        if (ord == 0) return
        (
            m_vf64A[idx]
            + m_vf64B[idx] * dx
            + m_vf64C[idx] * dx * dx
            + m_vf64D[idx] * dx * dx * dx
        );
        if (ord == 1) return
        (
            m_vf64B[idx]
            + m_vf64C[idx] * 2e0 * dx
            + m_vf64D[idx] * 3e0 * dx * dx
        );
        if (ord == 2) return
        (
            m_vf64C[idx] * 2e0
            + m_vf64D[idx] * 6e0 * dx
        );
        if (ord == 3) return
        (
            m_vf64D[idx] * 6e0
        );
        return 0e0;
    }

private:
    vf64 m_vf64A, m_vf64B, m_vf64C, m_vf64D;
};
