"""A numpy/stdlib stand-in for the handful of ``scipy.stats`` calls the web demo's
pipeline steps make, so the browser never downloads (13 MB) or imports (~2 s) scipy.

Covered: ``norm.ppf/pdf/cdf``, ``chi2.cdf`` (any df via the regularized lower gamma),
``skew``/``kurtosis`` (incl. ``bias=False``) and ``linregress`` (with the two-sided
t-test p-value). Anything else raises AttributeError, which the pipeline reports as a
warning. tests/test_web_demo.py checks every function against real scipy and runs
the demo pipeline both ways.
"""

from __future__ import annotations

import math
from collections import namedtuple
from statistics import NormalDist

import numpy as np

_N = NormalDist()


class _Norm:
    @staticmethod
    def ppf(q):
        q = np.asarray(q, dtype=float)
        f = np.vectorize(lambda v: _N.inv_cdf(v) if 0.0 < v < 1.0
                         else (-math.inf if v == 0.0 else math.inf if v == 1.0 else math.nan),
                         otypes=[float])
        out = f(q)
        return float(out) if out.ndim == 0 else out

    @staticmethod
    def pdf(x):
        x = np.asarray(x, dtype=float)
        out = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        return float(out) if out.ndim == 0 else out

    @staticmethod
    def cdf(x):
        x = np.asarray(x, dtype=float)
        out = 0.5 * (1.0 + np.vectorize(math.erf, otypes=[float])(x / math.sqrt(2.0)))
        return float(out) if out.ndim == 0 else out


def _gammainc_lower(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) (series / continued fraction)."""
    if x <= 0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return s * math.exp(-x + a * math.log(x) - gln)
    b, c, d = x + 1.0 - a, 1.0 / 1e-300, 1.0 / (x + 1.0 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - gln) * h


class _Chi2:
    @staticmethod
    def cdf(x, df):
        return _gammainc_lower(df / 2.0, float(x) / 2.0)

    @staticmethod
    def sf(x, df):
        return 1.0 - _Chi2.cdf(x, df)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) (Lentz continued fraction)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (tiny if abs(d) < tiny else d)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        num = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + num * d; d = 1.0 / (tiny if abs(d) < tiny else d)
        c = 1.0 + num / c; c = tiny if abs(c) < tiny else c
        h *= d * c
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + num * d; d = 1.0 / (tiny if abs(d) < tiny else d)
        c = 1.0 + num / c; c = tiny if abs(c) < tiny else c
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return front * h / a


def _t_two_sided_p(t: float, df: float) -> float:
    if not math.isfinite(t):
        return 0.0 if not math.isnan(t) else math.nan
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def _moments(a):
    x = np.asarray(a, dtype=float).ravel()
    x = x[~np.isnan(x)] if np.isnan(x).any() else x
    n = x.size
    d = x - x.mean()
    m2 = float(np.mean(d ** 2))
    return x, n, d, m2


def skew(a, bias: bool = True):
    x, n, d, m2 = _moments(a)
    if n < 3 or m2 == 0:
        return math.nan
    g1 = float(np.mean(d ** 3)) / m2 ** 1.5
    return g1 if bias else g1 * math.sqrt(n * (n - 1)) / (n - 2)


def kurtosis(a, fisher: bool = True, bias: bool = True):
    x, n, d, m2 = _moments(a)
    if n < 4 or m2 == 0:
        return math.nan
    g2 = float(np.mean(d ** 4)) / m2 ** 2 - 3.0
    if not bias:
        g2 = ((n + 1) * g2 + 6.0) * (n - 1) / ((n - 2) * (n - 3))
    return g2 if fisher else g2 + 3.0


LinregressResult = namedtuple("LinregressResult", "slope intercept rvalue pvalue stderr")


def linregress(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    xm, ym = x.mean(), y.mean()
    ssxm = float(np.mean((x - xm) ** 2))
    ssym = float(np.mean((y - ym) ** 2))
    ssxym = float(np.mean((x - xm) * (y - ym)))
    r = 0.0 if ssxm == 0 or ssym == 0 else max(-1.0, min(1.0, ssxym / math.sqrt(ssxm * ssym)))
    slope = ssxym / ssxm if ssxm else math.nan
    intercept = ym - slope * xm
    df = n - 2
    if n == 2:
        p, se = 0.0, 0.0
    else:
        t = r * math.sqrt(df / ((1.0 - r + 1e-300) * (1.0 + r + 1e-300)))
        p = _t_two_sided_p(t, df)
        se = math.sqrt((1 - r * r) * ssym / ssxm / df) if ssxm else math.nan
    return LinregressResult(slope, intercept, r, p, se)


norm = _Norm()
chi2 = _Chi2()


def install() -> None:
    """Expose this module as ``scipy.stats`` when real scipy is not installed."""
    import sys
    import types

    if "scipy" not in sys.modules:
        pkg = types.ModuleType("scipy")
        pkg.__path__ = []  # a namespace-like package: other submodules fail to import
        pkg.__doc__ = "web demo stand-in; only scipy.stats (pa_web.stats_lite) is provided"
        sys.modules["scipy"] = pkg
    mod = sys.modules[__name__]
    sys.modules["scipy.stats"] = mod
    sys.modules["scipy"].stats = mod
