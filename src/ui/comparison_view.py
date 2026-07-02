"""Multi-Portfolio Comparison — a distinct in-app section (a QStackedWidget page).

Left: a picker of saved portfolios plus the current working portfolio, and a Compare
button. Right: a WebTab-based report overlaying growth/drawdown, comparing risk/return
metrics, correlation, allocation/concentration, and holdings overlap. The comparison
runs the fast headless engine (``src/analytics/comparison.py``) on a background thread.
"""

from __future__ import annotations

import html as _html
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.charts import plotly_charts as charts
from src.config.models import PortfolioConfig

from . import paths, theme
from .tabs.web_tab import WebTab
from .worker import ComparisonWorker


CURRENT_LABEL = "Current working portfolio"


def _pct(v, d=1):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v * 100:.{d}f}%"


def _ratio(v, d=2):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.{d}f}"


class _ComparisonReport(WebTab):
    """Renders a list of ComparisonResult. Reuses all WebTab building blocks."""

    def show_results(self, results: list) -> None:
        self._results = results
        self.mark_dirty()
        self.ensure_populated()

    def _render_blank(self) -> None:
        t = theme.ACTIVE
        self._view.setHtml(
            f"<html><head>{self._css()}</head><body>"
            f"<div class='empty'>Select two or more portfolios on the left, then click "
            f"<b>Compare</b>.</div></body></html>",
            self._base_url,
        )

    def _populate(self, results) -> None:
        valid = [r for r in (results or []) if r.error is None and not r.values.empty]
        if len(valid) < 2:
            self.add_interpretation(
                "Select at least two portfolios that have price data, then click Compare."
            )
            errs = [r for r in (results or []) if r.error]
            for r in errs:
                self.add_interpretation(f"{r.label}: {r.error}")
            return

        self._render_performance(valid)
        self._render_risk(valid)
        self._render_allocation(valid)
        self._render_overlap(valid)

    # ── Performance ──
    def _render_performance(self, rs) -> None:
        self.add_zone("Performance", "Growth and risk-adjusted return")
        rebased = {}
        for r in rs:
            base = r.values.iloc[0]
            rebased[r.label] = (r.values / base * 1_000_000) if base else r.values
        self.add_heading("Growth of $1,000,000 (rebased)", explain=None)
        self.add_chart(charts.growth_chart(rebased, capital=1_000_000), height=420)

        self.add_heading("Return & Risk")
        self.add_table(pd.DataFrame({
            "Portfolio": [r.label for r in rs],
            "Ann. Return": [_pct(r.ann_return) for r in rs],
            "Volatility": [_pct(r.ann_vol) for r in rs],
            "Sharpe": [_ratio(r.sharpe) for r in rs],
            "Sortino": [_ratio(r.sortino) for r in rs],
            "Max Drawdown": [_pct(r.max_dd) for r in rs],
            "Alpha (ann.)": [_pct(r.alpha) for r in rs],
            "Beta": [_ratio(r.beta) for r in rs],
        }))

    # ── Risk ──
    def _render_risk(self, rs) -> None:
        from src.analytics.comparison import returns_correlation

        self.add_zone("Risk", "Drawdowns, tail risk, and how correlated they are")
        self.add_heading("Drawdown", explain="drawdown_chart")
        self.add_chart(charts.drawdown_chart({r.label: r.values for r in rs}), height=380)

        self.add_heading("Tail Risk (daily)")
        self.add_table(pd.DataFrame({
            "Portfolio": [r.label for r in rs],
            "VaR 95%": [_pct(r.var95, 2) for r in rs],
            "CVaR 95%": [_pct(r.cvar95, 2) for r in rs],
            "Max Drawdown": [_pct(r.max_dd) for r in rs],
        }))

        corr = returns_correlation(rs)
        if not corr.empty:
            self.add_heading("Return Correlation", explain="correlation_heatmap")
            self.add_chart(charts.correlation_heatmap(corr), height=360)

    # ── Allocation & exposure ──
    def _render_allocation(self, rs) -> None:
        self.add_zone("Allocation & Exposure", "How differently they're built")
        self.add_heading("Concentration", explain="concentration")
        self.add_table(pd.DataFrame({
            "Portfolio": [r.label for r in rs],
            "Holdings": [str(len(r.tickers)) for r in rs],
            "HHI": [_ratio(r.hhi, 4) for r in rs],
            "Effective N": [_ratio(r.effective_n, 1) for r in rs],
            "Top-3 Weight": [_pct(r.top3) for r in rs],
        }))

        if any(r.sector_weights for r in rs):
            sectors = sorted({s for r in rs if r.sector_weights for s in r.sector_weights})
            data = {"Portfolio": [r.label for r in rs]}
            for sec in sectors:
                data[sec] = [_pct((r.sector_weights or {}).get(sec, 0.0)) for r in rs]
            self.add_heading("Sector Weights")
            self.add_table(pd.DataFrame(data))

    # ── Holdings overlap ──
    def _render_overlap(self, rs) -> None:
        self.add_zone("Holdings Overlap", "Shared vs unique positions")
        sets = [set(r.tickers) for r in rs]
        shared = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        self.add_interpretation(
            f"{len(shared)} ticker(s) held by all {len(rs)} portfolios; "
            f"{len(union)} unique across the set."
        )
        self.add_heading("Weights by Holding")
        rows = sorted(union)
        data = {"Ticker": rows}
        for r in rs:
            w = r.weights or {}
            data[r.label] = [(_pct(w[t]) if t in w else "—") for t in rows]
        self.add_table(pd.DataFrame(data))


class ComparisonView(QWidget):
    backRequested = Signal()

    def __init__(self, get_current_config: Optional[Callable[[], PortfolioConfig]] = None,
                 parent=None) -> None:
        super().__init__(parent)
        self._get_current_config = get_current_config
        self._fetching = False
        self._thread: Optional[QThread] = None
        self._worker: Optional[ComparisonWorker] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 8)
        root.setSpacing(10)

        # Top bar
        top = QHBoxLayout()
        self._back_btn = QPushButton("←  Back to Analysis")
        self._back_btn.setCursor(Qt.PointingHandCursor)
        self._back_btn.clicked.connect(self.backRequested.emit)
        self._title = QLabel("Compare Portfolios")
        self._title.setObjectName("compareTitle")
        self._status = QLabel("")
        top.addWidget(self._back_btn)
        top.addSpacing(12)
        top.addWidget(self._title)
        top.addStretch(1)
        top.addWidget(self._status)
        root.addLayout(top)

        # Body: left picker + right report
        body = QHBoxLayout()
        body.setSpacing(12)
        left = QVBoxLayout()
        left.setSpacing(8)
        left.addWidget(QLabel("Portfolios to compare"))
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.NoSelection)
        self._list.setFixedWidth(240)
        left.addWidget(self._list, 1)
        self._compare_btn = QPushButton("Compare")
        self._compare_btn.setCursor(Qt.PointingHandCursor)
        self._compare_btn.clicked.connect(self._on_compare)
        left.addWidget(self._compare_btn)
        body.addLayout(left)

        self._report = _ComparisonReport()
        body.addWidget(self._report, 1)
        root.addLayout(body, 1)

        self.refresh_portfolio_list()

    # ── Portfolio list ──
    def refresh_portfolio_list(self) -> None:
        """Rebuild the checkable list from saved portfolios + the current one."""
        prev = self._checked_labels()
        self._list.clear()

        cur = QListWidgetItem(CURRENT_LABEL)
        cur.setFlags(cur.flags() | Qt.ItemIsUserCheckable)
        cur.setCheckState(Qt.Checked if (not prev or CURRENT_LABEL in prev) else Qt.Unchecked)
        cur.setData(Qt.UserRole, None)  # None => build from current config
        self._list.addItem(cur)

        try:
            files = sorted(paths.portfolios_dir().glob("*.json"))
        except Exception:
            files = []
        for f in files:
            item = QListWidgetItem(f.stem)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if f.stem in prev else Qt.Unchecked)
            item.setData(Qt.UserRole, str(f))
            self._list.addItem(item)

    def _checked_labels(self) -> set:
        out = set()
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.checkState() == Qt.Checked:
                out.add(it.text())
        return out

    # ── Run ──
    def _on_compare(self) -> None:
        if self._fetching:
            return
        named = []
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.checkState() != Qt.Checked:
                continue
            path = it.data(Qt.UserRole)
            try:
                if path is None:
                    if self._get_current_config is None:
                        continue
                    cfg = self._get_current_config()
                else:
                    cfg = PortfolioConfig.load(path)
            except Exception as e:
                self._set_status(f"{it.text()}: {e}")
                return
            named.append((it.text(), cfg))

        if len(named) < 2:
            self._set_status("Select at least two portfolios.")
            return
        if len(named) > 6:
            self._set_status("Compare up to 6 portfolios at a time.")
            return

        self._fetching = True
        self._compare_btn.setEnabled(False)
        self._set_status("Comparing…")
        self._thread = QThread(self)
        self._worker = ComparisonWorker(named)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.done.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()

    def _on_progress(self, label: str, frac: float) -> None:
        self._set_status(f"{label}  ({frac * 100:.0f}%)")

    def _on_done(self, results) -> None:
        self._fetching = False
        self._compare_btn.setEnabled(True)
        stamp = datetime.now(timezone.utc).astimezone().strftime("%I:%M %p").lstrip("0")
        n = len([r for r in results if r.error is None])
        self._set_status(f"Compared {n} portfolios  ·  {stamp}")
        self._report.show_results(results)

    def _on_failed(self, message: str) -> None:
        self._fetching = False
        self._compare_btn.setEnabled(True)
        self._set_status(f"Comparison failed: {message}")

    def _set_status(self, text: str) -> None:
        t = theme.ACTIVE
        self._status.setStyleSheet(f"color:{t.text_muted};font-size:{t.base_pt - 1}px;")
        self._status.setText(text)

    def retheme(self) -> None:
        t = theme.ACTIVE
        self._title.setStyleSheet(
            f"color:{t.text};font-size:{t.heading_pt + 2}px;font-weight:700;"
        )
        self._set_status(self._status.text())
        self._report.mark_dirty()
        self._report.ensure_populated()
