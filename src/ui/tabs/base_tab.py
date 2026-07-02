"""Base class and building blocks for result tabs.

A tab is a scrollable column of content. Subclasses implement ``_populate`` and
call the ``add_*`` helpers to append interpretation cards, headings, Plotly
charts, tables, and compact metric grids — the native equivalents of the
Streamlit calls in ``app.py``.

Population is lazy: ``set_results`` only marks the tab dirty; ``ensure_populated``
(driven by the tab widget's ``currentChanged``) rebuilds it the first time it is
shown for a given result set. This keeps ~40 QWebEngineViews from all spawning
Chromium processes at once.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)


class _AutoHeightTable(QTableView):
    """A table that reports its full content height, so it lays out completely
    (no internal scrolling) and the tab's single scroll area handles overflow."""

    def _content_height(self) -> int:
        length = self.verticalHeader().length()  # total height of all rows
        header = self.horizontalHeader().height()
        return length + header + 2 * self.frameWidth() + 2

    def sizeHint(self) -> QSize:
        return QSize(super().sizeHint().width(), self._content_height())

    def minimumSizeHint(self) -> QSize:
        return QSize(super().minimumSizeHint().width(), self._content_height())

from .. import explanations
from .. import theme
from ..widgets.dataframe_model import PandasTableModel
from ..widgets.info_label import InfoLabel
from ..widgets.plotly_widget import PlotlyWidget


class StatCard(QFrame):
    """A compact label/value tile for metric grids (Sortino, Calmar, ...)."""

    def __init__(self, label: str, value: str) -> None:
        super().__init__()
        t = theme.ACTIVE
        self.setStyleSheet(
            f"StatCard {{ background-color: {t.card}; border: 1px solid {t.border_light};"
            f" border-radius: {max(6, t.radius - 2)}px; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        lbl = QLabel(label.upper())
        lbl.setStyleSheet(
            f"color: {t.text_muted}; font-size: {t.label_pt - 1}px; font-weight: 700;"
            " letter-spacing: 0.05em; background: transparent;"
        )
        val = QLabel(value)
        val.setStyleSheet(
            f"color: {t.text}; font-size: {t.statval_pt}px; font-weight: 700;"
            f" font-family: {t.mono}; background: transparent;"
        )
        lay.addWidget(lbl)
        lay.addWidget(val)


class BaseTab(QScrollArea):
    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        # One scrollbar for the whole tab, on the right. Nested charts/tables
        # never scroll internally — they lay out at full size.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self._body = QWidget()
        self.content = QVBoxLayout(self._body)
        pad = theme.ACTIVE.content_spacing
        self.content.setContentsMargins(pad + 4, pad, pad + 4, pad + 10)
        self.content.setSpacing(pad)
        self.content.addStretch(1)  # trailing stretch, kept last
        self.setWidget(self._body)

        self._results = None
        self._dirty = True
        self._empty = QLabel("Run an analysis to populate this tab.")
        self._empty.setObjectName("muted")
        self._empty.setAlignment(Qt.AlignCenter)
        self.content.insertWidget(0, self._empty)

        # Chart views (embedded browsers) are expensive to create — pool and
        # reuse them across rebuilds instead of destroying/recreating, which
        # otherwise reinitializes Chromium and blanks the whole tab on each run.
        self._chart_pool: list[PlotlyWidget] = []
        self._active_charts: list[PlotlyWidget] = []

    # ── Lazy population ──
    def set_results(self, results) -> None:
        self._results = results
        self._dirty = True

    def mark_dirty(self) -> None:
        """Force a rebuild on next show (used after a theme switch)."""
        self._dirty = True

    def ensure_populated(self) -> None:
        if not self._dirty:
            return
        # Batch the teardown + rebuild into a single repaint so results appear in
        # one smooth pass rather than flickering widget-by-widget.
        self._body.setUpdatesEnabled(False)
        try:
            self._clear()
            if self._results is None:
                self._empty.setVisible(True)
                self.content.insertWidget(0, self._empty)
            else:
                self._empty.setVisible(False)
                self._populate(self._results)
        finally:
            self._body.setUpdatesEnabled(True)
        self._dirty = False

    def _populate(self, results) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    # ── Content clearing ──
    def _clear(self) -> None:
        # Detach live chart views first so they survive the teardown and can be
        # reused (keeps their Chromium process alive → no blank/relaunch flash).
        for pw in self._active_charts:
            pw.setParent(None)
            pw.hide()
            self._chart_pool.append(pw)
        self._active_charts.clear()

        # Delete everything else (labels, tables, chart holders) except the stretch.
        while self.content.count() > 1:
            item = self.content.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._empty:
                w.deleteLater()

    def prewarm(self, n: int = 4) -> None:
        """Create chart views ahead of time so the first run reuses them instead
        of creating (and initializing Chromium for) them mid-run."""
        while len(self._chart_pool) < n:
            pw = PlotlyWidget()
            pw.hide()
            self._chart_pool.append(pw)

    def _take_chart(self) -> PlotlyWidget:
        """Return a reusable chart view from the pool, or a fresh one."""
        pw = self._chart_pool.pop() if self._chart_pool else PlotlyWidget()
        pw.setMinimumHeight(0)
        pw.setMaximumHeight(16777215)  # clear any previous fixed height
        pw.show()
        self._active_charts.append(pw)
        return pw

    # ── Insertion helper (keeps the trailing stretch last) ──
    def _add(self, widget: QWidget) -> None:
        self.content.insertWidget(self.content.count() - 1, widget)

    def add_widget(self, widget: QWidget) -> None:
        """Append an arbitrary widget (buttons, custom panels)."""
        self._add(widget)

    # ── Building blocks (mirror app.py's st.* calls) ──
    def add_heading(self, text: str, explain: Optional[str] = None) -> None:
        t = theme.ACTIVE
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {t.text}; font-size: {t.heading_pt}px; font-weight: 700; margin-top: 6px;"
        )
        if explain and explanations.get(explain):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            h.addWidget(lbl)
            h.addWidget(InfoLabel(explanations.tooltip_html(explain)))
            h.addStretch(1)
            self._add(row)
        else:
            self._add(lbl)

    def add_interpretation(self, text: Optional[str]) -> None:
        if not text:
            return
        t = theme.ACTIVE
        card = QLabel(text)
        card.setWordWrap(True)
        card.setStyleSheet(
            f"background: {t.card}; border-left: 3px solid {t.accent};"
            f" padding: 14px 18px; border-radius: 0 8px 8px 0; color: {t.text_slate};"
            f" font-size: {t.base_pt}px; line-height: 1.5;"
        )
        self._add(card)

    def _scaled(self, height: int) -> int:
        return int(height * theme.ACTIVE.chart_scale)

    def add_explainer(self, key: Optional[str]) -> None:
        """Add a plain-English explanation affordance for a result.

        Always shows a subtle 'ⓘ What is this?' with a full-text tooltip; in
        Beginner mode it expands to an inline blurb so nothing is hidden.
        """
        from .. import explanations

        if not key or explanations.get(key) is None:
            return
        t = theme.ACTIVE
        if explanations.is_beginner_mode():
            e = explanations.get(key)
            lbl = QLabel(f"<b>{e['title']}.</b> {explanations.inline_text(key)}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(
                f"color: {t.text_slate}; font-size: {t.base_pt - 1}px;"
                f" background: {t.card}; border-radius: {max(4, t.radius - 4)}px;"
                " padding: 8px 12px;"
            )
            self._add(lbl)
        else:
            self._add(InfoLabel(explanations.tooltip_html(key)))

    def add_chart(self, fig, height: int = 420, explain: Optional[str] = None) -> None:
        if fig is None:
            return
        self.add_explainer(explain)
        pw = self._take_chart()
        pw.setFixedHeight(self._scaled(height))
        pw.set_figure(fig)
        self._add(pw)

    def add_chart_row(self, figs: list, height: int = 380, explain: Optional[str] = None) -> None:
        """Place two or more charts side by side in one row."""
        figs = [f for f in figs if f is not None]
        if not figs:
            return
        self.add_explainer(explain)
        if len(figs) == 1:
            self.add_chart(figs[0], height=height)
            return
        holder = QWidget()
        holder.setFixedHeight(self._scaled(height))
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        for fig in figs:
            pw = self._take_chart()
            pw.set_figure(fig)
            row.addWidget(pw, stretch=1)
        self._add(holder)

    def add_table(
        self,
        df: pd.DataFrame,
        formatters: Optional[dict[str, Callable]] = None,
        show_index: bool = False,
    ) -> None:
        if df is None or len(df) == 0:
            return
        frame = df if show_index else df.reset_index(drop=True)
        view = _AutoHeightTable()
        model = PandasTableModel(frame, formatters=formatters)
        view.setModel(model)
        view.setAlternatingRowColors(True)
        view.verticalHeader().setVisible(False)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        view.setEditTriggers(QTableView.NoEditTriggers)
        view.setSelectionMode(QTableView.NoSelection)
        # Never scroll internally — the tab's single scroll area owns scrolling.
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Size rows to their (possibly larger, scaled) font, then let the table
        # report its full content height via sizeHint (vertical policy = Fixed).
        view.resizeRowsToContents()
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._add(view)

    def add_chart_grid(
        self, figs: list, columns: int = 3, height: int = 300, explain: Optional[str] = None
    ) -> None:
        """Lay out many small charts in a grid (e.g. CAPM scatters)."""
        figs = [f for f in figs if f is not None]
        if not figs:
            return
        self.add_explainer(explain)
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)
        for i, fig in enumerate(figs):
            pw = self._take_chart()
            pw.setFixedHeight(self._scaled(height))
            pw.set_figure(fig)
            grid.addWidget(pw, i // columns, i % columns)
        self._add(holder)

    def add_stat_grid(self, items: list[tuple[str, str]], columns: int = 4) -> None:
        """A grid of compact label/value StatCards."""
        if not items:
            return
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        for i, (label, value) in enumerate(items):
            grid.addWidget(StatCard(label, value), i // columns, i % columns)
        self._add(holder)
