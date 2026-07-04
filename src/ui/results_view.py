"""The results area: a tab widget that maps ``AnalysisResults`` onto the result
tabs, populating each lazily on first view.

The chart tabs are ``WebTab``s (one HTML page + one web view each); the Data tab is
a Qt ``BaseTab``. All expose the same set_results/mark_dirty/ensure_populated/prewarm
interface, so this view treats them uniformly.

The **Macro** tab is special: it only appears once a valid FRED API key returns data,
and is inserted/removed dynamically based on the tab's ``availabilityChanged`` signal.
"""

from __future__ import annotations

from PySide6.QtWidgets import QTabWidget

from .tabs.attribution_tab import AttributionTab
from .tabs.data_tab import DataTab
from .tabs.forecast_tab import ForecastTab
from .tabs.fundamentals_tab import FundamentalsTab
from .tabs.income_tab import IncomeTab
from .tabs.macro_tab import MacroTab
from .tabs.news_tab import NewsTab
from .tabs.optimization_tab import OptimizationTab
from .tabs.overview_tab import OverviewTab
from .tabs.performance_tab import PerformanceTab
from .tabs.risk_tab import RiskTab


class ResultsView(QTabWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setDocumentMode(True)

        self._news_tab = NewsTab()
        self._tabs = [
            OverviewTab(),
            PerformanceTab(),
            RiskTab(),
            AttributionTab(),
            IncomeTab(),
            OptimizationTab(),
            ForecastTab(),
            FundamentalsTab(),
            self._news_tab,
            DataTab(),
        ]
        labels = ["Overview", "Performance", "Risk", "Attribution", "Income",
                  "Optimization", "Forecast", "Fundamentals", "News", "Data"]
        for tab, label in zip(self._tabs, labels):
            self.addTab(tab, label)

        # Macro tab is added/removed dynamically based on FRED-key availability.
        self._macro_tab = MacroTab()
        self._macro_visible = False
        self._macro_tab.availabilityChanged.connect(self._sync_macro_tab)

        self.currentChanged.connect(self._on_tab_changed)

    def display(self, results) -> None:
        """Push a new result set to every tab and eagerly render ALL of them, so
        switching tabs afterward is instant (no per-tab load)."""
        for tab in self._tabs:
            tab.set_results(results)
        self._macro_tab.set_results(results)  # may reveal the Macro tab via its signal
        self._on_tab_changed(self.currentIndex())
        for tab in self._tabs:
            tab.ensure_populated()

    def prewarm(self) -> None:
        return

    def refresh_news(self) -> None:
        """Re-fetch News and Macro (e.g. after API keys change)."""
        self._news_tab.refresh()
        self._macro_tab.refresh()

    def _sync_macro_tab(self, available: bool) -> None:
        if available and not self._macro_visible:
            index = self.indexOf(self._news_tab) + 1
            self.insertTab(index, self._macro_tab, "Macro")
            self._macro_visible = True
        elif not available and self._macro_visible:
            self.removeTab(self.indexOf(self._macro_tab))
            self._macro_visible = False

    def retheme(self) -> None:
        """Rebuild tabs under the active theme; repaint the visible one now."""
        for tab in self._tabs:
            tab.mark_dirty()
        self._macro_tab.mark_dirty()
        self._on_tab_changed(self.currentIndex())

    def _on_tab_changed(self, index: int) -> None:
        widget = self.widget(index)
        if hasattr(widget, "ensure_populated"):
            widget.ensure_populated()

    def shutdown(self) -> None:
        """Stop any tab's background fetch threads cleanly (called on app close)."""
        for tab in (*self._tabs, self._macro_tab):
            fn = getattr(tab, "shutdown", None)
            if callable(fn):
                fn()
