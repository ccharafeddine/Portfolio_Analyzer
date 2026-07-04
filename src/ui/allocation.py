"""Share-count → weights + capital conversion.

Qt-free (unit-tested on CI). Turns "I hold N shares of each" plus current prices
into the portfolio's market value (capital) and value-based weights, which is
what the analysis pipeline actually consumes.
"""

from __future__ import annotations


def active_allocation_weights(weights: dict, capital: float, cash: float) -> dict:
    """Return the actual account allocation including a Cash slice.

    ``weights`` are the risky (stock) weights summing to 1; ``capital`` is the
    TOTAL account value and ``cash`` the cash balance (so invested = capital -
    cash). Stock weights are scaled by the risky fraction ``y = invested / total``
    and a ``"Cash"`` slice of ``cash / total`` is added, so the result sums to 1
    over the whole account. With no cash, the weights pass through unchanged.
    """
    total = float(capital or 0.0)
    csh = float(cash or 0.0)
    if csh <= 0 or total <= 0:
        return {t: float(w) for t, w in (weights or {}).items()}
    invested = max(total - csh, 0.0)
    y = invested / total
    out = {t: float(w) * y for t, w in (weights or {}).items()}
    out["Cash"] = csh / total
    return out


def shares_to_weights_and_capital(shares: dict, prices: dict) -> tuple[dict, float]:
    """Return ``(weights, capital)`` from share counts and per-share prices.

    - ``capital`` is the total market value Σ shares×price over holdings that
      have a positive share count *and* a positive price.
    - ``weights`` are value-based and sum to 1 (``value_i / capital``).
    Holdings missing a price (or with non-positive shares) are skipped; if
    nothing can be priced, returns ``({}, 0.0)``.
    """
    values: dict[str, float] = {}
    for ticker, qty in (shares or {}).items():
        t = str(ticker).strip().upper()
        try:
            q = float(qty)
            p = float(prices.get(t) or prices.get(ticker) or 0.0)
        except (TypeError, ValueError):
            continue
        if q > 0 and p > 0:
            values[t] = q * p

    capital = sum(values.values())
    if capital <= 0:
        return {}, 0.0
    weights = {t: v / capital for t, v in values.items()}
    return weights, capital
