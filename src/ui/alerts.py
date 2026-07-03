"""Price alerts: model, edge-triggered evaluation, and persistence.

Qt-free at import (unit-testable on CI's headless runner). ``QSettings`` is
imported lazily inside :class:`AlertStore` persistence only. The pure
:func:`evaluate` function is what the main window calls on each quotes snapshot.

An alert fires once when the price *crosses* its threshold (edge-triggered), not
on every poll while the condition holds, and re-arms when the price moves back to
the other side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


def normalize_direction(d: str) -> str:
    return "above" if str(d).strip().lower().startswith("a") else "below"


@dataclass
class Alert:
    ticker: str
    direction: str            # "above" | "below"
    price: float
    enabled: bool = True
    # Runtime edge-detection state (whether the condition held last poll). Not
    # persisted; ``None`` means "not yet observed" so the first poll only seeds.
    last_met: Optional[bool] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        self.ticker = str(self.ticker).strip().upper()
        self.direction = normalize_direction(self.direction)
        self.price = float(self.price)

    def condition_met(self, last: float) -> bool:
        return last >= self.price if self.direction == "above" else last <= self.price

    def describe(self) -> str:
        arrow = "≥" if self.direction == "above" else "≤"
        return f"{self.ticker} {arrow} {self.price:,.2f}"

    def to_dict(self) -> dict:
        return {"ticker": self.ticker, "direction": self.direction,
                "price": self.price, "enabled": self.enabled}

    @classmethod
    def from_dict(cls, d: dict) -> "Alert":
        return cls(ticker=d.get("ticker", ""), direction=d.get("direction", "above"),
                   price=float(d.get("price", 0.0)), enabled=bool(d.get("enabled", True)))


def evaluate(alerts, quotes) -> list[Alert]:
    """Return the alerts that just crossed their threshold this snapshot.

    ``quotes`` is ``{ticker: quote}`` where ``quote.last`` is the latest price.
    Edge-triggered: an alert fires only on the transition into its condition, and
    its ``last_met`` state is updated in place so repeat polls don't re-fire.
    """
    triggered: list[Alert] = []
    for a in alerts:
        if not a.enabled:
            continue
        q = quotes.get(a.ticker)
        last = getattr(q, "last", None)
        if not isinstance(last, (int, float)) or last != last:
            continue  # no price this poll; leave state untouched
        met = a.condition_met(last)
        if a.last_met is None:
            a.last_met = met       # seed on first observation; never fire immediately
        elif met and not a.last_met:
            triggered.append(a)
        a.last_met = met
    return triggered


class AlertStore:
    """In-memory list of alerts with QSettings-backed persistence."""

    _KEY = "alerts"

    def __init__(self, autoload: bool = True) -> None:
        self._alerts: list[Alert] = []
        if autoload:
            self.load()

    # ── Access ──
    def all(self) -> list[Alert]:
        return list(self._alerts)

    def tickers(self) -> list[str]:
        """Distinct tickers of the *enabled* alerts (for the poll universe)."""
        return list(dict.fromkeys(a.ticker for a in self._alerts if a.enabled))

    # ── Mutation ──
    def add(self, ticker: str, direction: str, price: float) -> Alert:
        alert = Alert(ticker=ticker, direction=direction, price=price)
        self._alerts.append(alert)
        self.save()
        return alert

    def remove(self, index: int) -> None:
        if 0 <= index < len(self._alerts):
            del self._alerts[index]
            self.save()

    def set_enabled(self, index: int, enabled: bool) -> None:
        if 0 <= index < len(self._alerts):
            self._alerts[index].enabled = bool(enabled)
            self._alerts[index].last_met = None  # re-arm on toggle
            self.save()

    def clear(self) -> None:
        self._alerts = []
        self.save()

    def evaluate(self, quotes) -> list[Alert]:
        return evaluate(self._alerts, quotes)

    # ── Persistence (QSettings; lazy import keeps this module Qt-free) ──
    def _settings(self):
        from PySide6.QtCore import QSettings

        from .settings import APP_NAME, ORG_NAME

        return QSettings(ORG_NAME, APP_NAME)

    def load(self) -> None:
        try:
            raw = self._settings().value(self._KEY)
            data = json.loads(raw) if raw else []
            self._alerts = [Alert.from_dict(d) for d in data]
        except Exception:
            self._alerts = []

    def save(self) -> None:
        try:
            self._settings().setValue(
                self._KEY, json.dumps([a.to_dict() for a in self._alerts])
            )
        except Exception:
            pass
