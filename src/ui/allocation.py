"""Share-count → weights + capital conversion.

Qt-free (unit-tested on CI). Turns "I hold N shares of each" plus current prices
into the portfolio's market value (capital) and value-based weights, which is
what the analysis pipeline actually consumes.
"""

from __future__ import annotations


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
