"""Macro tab — Treasury curve and headline rates from FRED.

Only meaningful with a FRED API key, so the tab reveals itself (via
``availabilityChanged``) only after a key returns data, and hides when the key is
absent or invalid. Fetched on a background thread. Not included in exported reports.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from PySide6.QtCore import Signal

from src.charts import plotly_charts as charts

from .. import settings
from ..worker import MacroWorker
from .refreshable_tab import RefreshableWebTab


class MacroTab(RefreshableWebTab):
    # Emitted after a fetch: True when macro data is available (valid FRED key).
    availabilityChanged = Signal(bool)
    # Emitted with the latest short-term Treasury yield (as a decimal, e.g. 0.051)
    # and its tenor label (e.g. '3M') so the config can track the live risk-free rate.
    riskFreeRateReady = Signal(float, str)

    def __init__(self) -> None:
        super().__init__()
        self._macro = None
        self._last_updated: Optional[datetime] = None

    def refresh(self) -> None:
        if self._fetching:
            return
        if not settings.get_api_key("FRED_API_KEY"):
            self._macro = None
            self._set_status("Add a FRED key in Settings")
            self.availabilityChanged.emit(False)
            return
        self._set_status("Fetching macro data…")
        self._start(MacroWorker(settings.get_api_key("FRED_API_KEY")), self._on_macro)

    def _on_macro(self, macro) -> None:
        self._macro = macro
        available = macro is not None
        if available:
            self._last_updated = datetime.now(timezone.utc)
            stamp = self._last_updated.astimezone().strftime("%I:%M %p").lstrip("0")
            self._set_status(f"Updated {stamp}")
            self.mark_dirty()
            self.ensure_populated()
            rf = self._risk_free_from_curve(macro.curve)
            if rf is not None:
                self.riskFreeRateReady.emit(rf[0], rf[1])
        else:
            self._set_status("No macro data (check your FRED key)")
        self.availabilityChanged.emit(available)

    @staticmethod
    def _risk_free_from_curve(curve: dict):
        """The shortest-tenor yield as ``(decimal_rate, tenor_label)`` — the
        risk-free proxy. Returns None if the curve is empty."""
        if not curve:
            return None
        for tenor in ("3M", "6M", "1Y", "2Y"):
            if tenor in curve:
                return float(curve[tenor]) / 100.0, tenor
        tenor, val = next(iter(curve.items()))
        return float(val) / 100.0, tenor

    # ── Rendering ──
    def _populate(self, results) -> None:
        if self._macro is None:
            self.add_heading("Rates & Treasuries", explain="rates_macro")
            self.add_interpretation(
                "Add a free FRED API key in Settings → Preferences to load the "
                "Treasury yield curve and headline macro rates."
            )
            return

        self.add_heading("U.S. Treasury Yield Curve", explain="rates_macro")
        if self._macro.curve:
            self.add_chart(charts.treasury_curve_chart(self._macro.curve), height=360)
        if self._macro.rates:
            rows = []
            for r in self._macro.rates:
                chg = r.get("change_1y")
                rows.append({
                    "Indicator": r["name"],
                    "Latest": f"{r['value']:.2f}%",
                    "1-Yr Change": ("—" if chg is None else f"{chg:+.2f} pp"),
                    "As of": r["date"],
                })
            self.add_heading("Key Rates")
            self.add_table(pd.DataFrame(rows))
        if self._macro.series:
            label_map = {"DGS10": "10-Year Treasury", "FEDFUNDS": "Fed Funds Rate"}
            series = {label_map.get(k, k): v for k, v in self._macro.series.items()}
            self.add_heading("Rate History")
            self.add_chart(
                charts.rate_history_chart(series, "10-Year Treasury vs Fed Funds"),
                height=340,
            )
