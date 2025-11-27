
import numpy as np

# === Bond Math ===

def price_coupon_bond(face: float, coupon_rate: float, ytm: float, n_periods: int, freq: int = 2) -> float:
    c = coupon_rate * face / freq
    y = ytm / freq
    disc = (1 + y) ** np.arange(1, n_periods + 1)
    pv_coupons = np.sum(c / disc)
    pv_face = face / ((1 + y) ** n_periods)
    return float(pv_coupons + pv_face)

def ytm_from_price(face: float, coupon_rate: float, price: float, n_periods: int, freq: int = 2, tol: float = 1e-8, maxit: int = 200) -> float:
    ytm = coupon_rate if coupon_rate > 0 else 0.05
    for _ in range(maxit):
        c = coupon_rate * face / freq
        y = ytm / freq
        disc = (1 + y) ** np.arange(1, n_periods + 1)
        pv = np.sum(c / disc) + face / ((1 + y) ** n_periods)
        f = pv - price
        d_pv_d_y = np.sum(-np.arange(1, n_periods + 1) * c / (disc * (1 + y))) + (-n_periods) * face / (((1 + y) ** (n_periods + 1)))
        d_pv_d_ytm = d_pv_d_y / freq
        step = f / d_pv_d_ytm
        ytm -= step
        if abs(step) < tol:
            break
    return float(ytm)

def yield_to_call(price: float, face: float, coupon_rate: float, call_price: float, periods_to_call: int, freq: int = 2, tol: float = 1e-8, maxit: int = 200) -> float:
    ytc = coupon_rate if coupon_rate > 0 else 0.05
    for _ in range(maxit):
        c = coupon_rate * face / freq
        y = ytc / freq
        disc = (1 + y) ** np.arange(1, periods_to_call + 1)
        pv = np.sum(c / disc) + call_price / ((1 + y) ** periods_to_call)
        f = pv - price
        d_pv_d_y = np.sum(-np.arange(1, periods_to_call + 1) * c / (disc * (1 + y))) + (-periods_to_call) * call_price / (((1 + y) ** (periods_to_call + 1)))
        d_pv_d_ytc = d_pv_d_y / freq
        step = f / d_pv_d_ytc
        ytc -= step
        if abs(step) < tol:
            break
    return float(ytc)

# === Stock Valuation ===

def ddm_constant_growth(div1: float, r_e: float, g: float) -> float:
    if r_e <= g:
        raise ValueError("r_e must be > g in constant-growth DDM.")
    return float(div1 / (r_e - g))

def implied_cost_of_equity(price: float, div1: float, g: float) -> float:
    return float(div1 / price + g)


if __name__ == "__main__":
    pass
