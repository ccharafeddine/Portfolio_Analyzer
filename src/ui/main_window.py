"""Main application window and shell.

The window layout — menubar, config dock, headline metric strip, a 7-tab results
area, and a status bar with a progress bar. The config panel is bound to
``PortfolioConfig`` (Phase 2); the results tabs render ``AnalysisResults``
(Phase 3). Phase 4 wires the Run action to a threaded pipeline.
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, QThread, QSettings, QSize, QTimer
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from . import __app_name__, __version__
from . import explanations
from . import theme
from .assets import mark_path
from .config_panel import ConfigPanel
from .sidebar import Sidebar
from .formatting import delta_str, fmt_pct
from .results_view import ResultsView
from .settings import APP_NAME, ORG_NAME
from .widgets.metric_card import MetricStrip
from .worker import AnalysisWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(__app_name__)
        self.setWindowIcon(QIcon(mark_path()))
        self.resize(1360, 860)
        self.setMinimumSize(1024, 680)

        # Restore the saved theme + text scale before building any styled widgets.
        self._settings = QSettings(ORG_NAME, APP_NAME)
        saved = self._settings.value("theme")
        if saved in theme.THEMES:
            theme.set_active(saved)
        theme.set_scale(self._settings.value("ui_scale", theme.DEFAULT_SCALE, type=float))
        self._apply_chart_palette()
        self.setStyleSheet(theme.stylesheet())

        # Analysis thread state
        self._thread: QThread | None = None
        self._worker: AnalysisWorker | None = None
        self._cancelled = False
        self._last_results = None
        self._current_portfolio_path: str | None = None

        self._build_menubar()
        self._build_central()
        self._build_statusbar()

        self.act_cancel.triggered.connect(self._on_cancel)

        # Warm up chart views (and the QtWebEngine subsystem) right after the
        # window is shown, so the first Run doesn't flash while creating them.
        QTimer.singleShot(0, self.results_view.prewarm)

    # ── Menubar ──
    def _build_menubar(self) -> None:
        bar = self.menuBar()

        file_menu = bar.addMenu("&File")
        self.act_new = QAction("New Portfolio", self, shortcut=QKeySequence.New)
        self.act_open = QAction("Open Portfolio…", self, shortcut=QKeySequence.Open)
        self.act_save = QAction("Save Portfolio…", self, shortcut=QKeySequence.Save)
        self.act_import_csv = QAction("Import Holdings CSV…", self)
        self.act_export = QAction("Export Report…", self)
        self.act_quit = QAction("Exit", self, shortcut=QKeySequence.Quit)
        self.act_new.triggered.connect(self._on_new_portfolio)
        self.act_open.triggered.connect(self._on_open_portfolio)
        self.act_save.triggered.connect(self._on_save_portfolio)
        self.act_import_csv.triggered.connect(self._on_import_csv)
        self.act_export.triggered.connect(self._on_export_report)
        self.act_quit.triggered.connect(self.close)
        for a in (self.act_new, self.act_open, self.act_save):
            file_menu.addAction(a)
        file_menu.addSeparator()
        file_menu.addAction(self.act_import_csv)
        file_menu.addAction(self.act_export)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        run_menu = bar.addMenu("&Run")
        self.act_run = QAction("Run Analysis", self, shortcut="Ctrl+R")
        self.act_cancel = QAction("Cancel", self)
        self.act_cancel.setEnabled(False)
        run_menu.addAction(self.act_run)
        run_menu.addAction(self.act_cancel)

        # Top-level entry into the multi-portfolio comparison section.
        self.act_compare = QAction("Compare Portfolios", self)
        self.act_compare.triggered.connect(self._show_compare_mode)
        bar.addAction(self.act_compare)

        view_menu = bar.addMenu("&View")
        theme_menu = view_menu.addMenu("Theme")
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        self._theme_actions: dict[str, QAction] = {}
        for key, th in theme.THEMES.items():
            act = QAction(th.name, self, checkable=True)
            act.setChecked(key == theme.ACTIVE.key)
            act.triggered.connect(lambda _=False, k=key: self._apply_theme(k))
            self._theme_group.addAction(act)
            theme_menu.addAction(act)
            self._theme_actions[key] = act

        size_menu = view_menu.addMenu("Text Size")
        self._scale_group = QActionGroup(self)
        self._scale_group.setExclusive(True)
        self._scale_actions: list[tuple[float, QAction]] = []
        for label, factor in theme.SCALE_PRESETS:
            act = QAction(f"{label}  ({int(factor * 100)}%)", self, checkable=True)
            act.setChecked(abs(factor - theme.current_scale()) < 1e-6)
            act.triggered.connect(lambda _=False, f=factor: self._apply_scale(f))
            self._scale_group.addAction(act)
            size_menu.addAction(act)
            self._scale_actions.append((factor, act))
        size_menu.addSeparator()
        act_inc = QAction("Increase", self, shortcut=QKeySequence.ZoomIn)
        act_dec = QAction("Decrease", self, shortcut=QKeySequence.ZoomOut)
        act_reset = QAction("Reset", self, shortcut="Ctrl+0")
        act_inc.triggered.connect(lambda: self._apply_scale(theme.current_scale() + 0.1))
        act_dec.triggered.connect(lambda: self._apply_scale(theme.current_scale() - 0.1))
        act_reset.triggered.connect(lambda: self._apply_scale(theme.DEFAULT_SCALE))
        size_menu.addAction(act_inc)
        size_menu.addAction(act_dec)
        size_menu.addAction(act_reset)

        view_menu.addSeparator()
        self.act_toggle_sidebar = QAction("Toggle Sidebar", self, shortcut="Ctrl+B")
        self.act_toggle_sidebar.triggered.connect(lambda: self.sidebar.toggle())
        view_menu.addAction(self.act_toggle_sidebar)

        view_menu.addSeparator()
        self.act_beginner = QAction("Beginner mode (explanations)", self, checkable=True)
        self.act_beginner.setChecked(explanations.is_beginner_mode())
        self.act_beginner.toggled.connect(self._on_beginner_toggled)
        view_menu.addAction(self.act_beginner)

        settings_menu = bar.addMenu("&Settings")
        self.act_settings = QAction("Preferences…", self)
        self.act_settings.triggered.connect(self._show_settings)
        settings_menu.addAction(self.act_settings)

        help_menu = bar.addMenu("&Help")
        self.act_updates = QAction("Check for Updates…", self)
        self.act_about = QAction("About", self)
        self.act_about.triggered.connect(self._show_about)
        help_menu.addAction(self.act_updates)
        help_menu.addAction(self.act_about)

    # ── Central area: collapsible sidebar + (header, metrics, results) ──
    def _build_central(self) -> None:
        # Config panel lives in a collapsible left sidebar (not a floating dock).
        self.config_panel = ConfigPanel()
        self.config_panel.runRequested.connect(self._on_run_requested)
        self.sidebar = Sidebar(self.config_panel, title="Configuration")
        # Menu "Run Analysis" triggers the same path as the panel button.
        self.act_run.triggered.connect(self.config_panel._on_run)

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(16, 14, 16, 8)
        main_layout.setSpacing(14)
        main_layout.addWidget(self._build_header())
        self.metric_strip = MetricStrip()
        main_layout.addWidget(self.metric_strip)
        self.results_view = ResultsView()
        main_layout.addWidget(self.results_view, stretch=1)
        # When FRED macro loads, track the live short-term yield as the risk-free
        # rate for the next run.
        self.results_view._macro_tab.riskFreeRateReady.connect(self._on_risk_free_rate)

        analyze = QWidget()
        row = QHBoxLayout(analyze)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.sidebar)
        row.addWidget(main, stretch=1)

        # A stack switches the whole workspace between single-portfolio "Analyze"
        # mode and the multi-portfolio "Compare" section.
        from PySide6.QtWidgets import QStackedWidget

        from .comparison_view import ComparisonView

        self._stack = QStackedWidget()
        self._stack.addWidget(analyze)                 # page 0: Analyze
        self.comparison_view = ComparisonView(
            get_current_config=self.config_panel.build_config
        )
        self.comparison_view.backRequested.connect(self._show_analyze_mode)
        self._stack.addWidget(self.comparison_view)    # page 1: Compare
        self.setCentralWidget(self._stack)

    def _build_header(self) -> QWidget:
        header = QWidget()
        row = QHBoxLayout(header)
        row.setContentsMargins(2, 2, 2, 6)
        row.setSpacing(14)

        self._header_logo = QSvgWidget(mark_path())
        row.addWidget(self._header_logo)

        self._header_title = QLabel(__app_name__)
        row.addWidget(self._header_title)
        row.addStretch(1)
        self._retheme_header()
        return header

    def _retheme_header(self) -> None:
        t = theme.ACTIVE
        side = int(44 * theme.current_scale())
        self._header_logo.setFixedSize(QSize(side, side))
        self._header_title.setStyleSheet(
            f"color: {t.text}; font-size: {t.heading_pt + 10}px; font-weight: 800;"
            " letter-spacing: -0.01em; background: transparent;"
        )

    # ── Status bar with progress ──
    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        # An idle status label and a full-width progress bar share the bar; only
        # one is visible at a time, so the progress bar spans the whole bottom.
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        sb.addWidget(self.status_label, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setMinimumHeight(20)
        self.progress.setVisible(False)
        sb.addWidget(self.progress, 1)

    def _status(self, text: str) -> None:
        self.status_label.setText(text)

    # ── Theming ──
    def _apply_chart_palette(self) -> None:
        from src.charts import plotly_charts

        plotly_charts.apply_palette(**theme.chart_palette())

    def _restyle_all(self) -> None:
        """Re-apply the active theme + scale to every part of the window."""
        self._apply_chart_palette()
        self.setStyleSheet(theme.stylesheet())
        self._retheme_header()
        self.config_panel.retheme()
        self.metric_strip.retheme()
        if self._last_results is not None:
            self._update_metrics(self._last_results)
        self.results_view.retheme()
        if hasattr(self, "comparison_view"):
            self.comparison_view.retheme()

    # ── Mode switch (Analyze <-> Compare) ──
    def _show_compare_mode(self) -> None:
        self.comparison_view.refresh_portfolio_list()
        self._stack.setCurrentWidget(self.comparison_view)
        self._status("Compare portfolios — pick from the list and click Compare")

    def _show_analyze_mode(self) -> None:
        self._stack.setCurrentIndex(0)

    def _apply_theme(self, key: str) -> None:
        theme.set_active(key)
        self._restyle_all()
        if key in self._theme_actions:
            self._theme_actions[key].setChecked(True)
        self._settings.setValue("theme", key)
        self._status(f"Theme: {theme.ACTIVE.name}")

    def _apply_scale(self, factor: float) -> None:
        applied = theme.set_scale(factor)
        self._restyle_all()
        for f, act in self._scale_actions:
            act.setChecked(abs(f - applied) < 1e-6)
        self._settings.setValue("ui_scale", applied)
        self._status(f"Text size: {int(applied * 100)}%")

    # ── File: portfolio management ──
    def _on_new_portfolio(self) -> None:
        self.config_panel.reset_defaults()
        self._current_portfolio_path = None
        self._status("New portfolio — configuration reset to defaults")

    def _on_open_portfolio(self) -> None:
        from pathlib import Path

        from . import paths
        from .config_panel import ConfigPanel  # noqa: F401 (type hint clarity)
        from src.config.models import PortfolioConfig

        path, _ = QFileDialog.getOpenFileName(
            self, "Open Portfolio", str(paths.portfolios_dir()),
            "Portfolio files (*.json)",
        )
        if not path:
            return
        try:
            config = PortfolioConfig.load(path)
        except Exception:
            try:
                config = PortfolioConfig.from_legacy(path)  # tolerate older files
            except Exception as e:
                QMessageBox.warning(self, "Open Portfolio", f"Could not open this file:\n{e}")
                return
        self.config_panel.load_config(config)
        self._current_portfolio_path = path
        self._status(f"Opened {Path(path).name}")

    def _on_save_portfolio(self) -> None:
        from pathlib import Path

        from . import paths

        try:
            config = self.config_panel.build_config()
        except Exception as e:
            QMessageBox.warning(
                self, "Save Portfolio",
                f"Fix the configuration before saving:\n{self.config_panel._friendly_error(e)}",
            )
            return
        default = self._current_portfolio_path or str(paths.portfolios_dir() / "portfolio.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Portfolio", default, "Portfolio files (*.json)"
        )
        if not path:
            return
        try:
            config.save(path)
            self._current_portfolio_path = path
            self._status(f"Saved portfolio to {Path(path).name}")
        except Exception as e:
            QMessageBox.warning(self, "Save Portfolio", f"Could not save:\n{e}")

    def _on_import_csv(self) -> None:
        from pathlib import Path

        from . import paths

        path, _ = QFileDialog.getOpenFileName(
            self, "Import Holdings CSV", str(paths.documents_export_dir()),
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            n = self.config_panel.import_holdings_csv(path)
            self._status(f"Imported {n} holdings from {Path(path).name}")
        except Exception as e:
            QMessageBox.warning(self, "Import Holdings CSV", str(e))

    def _on_export_report(self) -> None:
        # Reports live on the Data tab; jump there so the user can pick a format.
        if self._last_results is None:
            self._status("Run an analysis first, then export from the Data tab")
            return
        for i in range(self.results_view.count()):
            if self.results_view.tabText(i) == "Data":
                self.results_view.setCurrentIndex(i)
                break
        self._status("Choose a report format on the Data tab")

    # ── Live risk-free rate from FRED ──
    def _on_risk_free_rate(self, rate: float, tenor: str) -> None:
        self.config_panel.set_risk_free_rate(rate, source=f"live {tenor} T-bill")
        self._status(f"Risk-free rate set to latest {tenor} T-bill yield ({rate:.2%})")

    # ── Preferences ──
    def _show_settings(self) -> None:
        from .settings_dialog import SettingsDialog

        was_beginner = explanations.is_beginner_mode()
        dialog = SettingsDialog(self)
        if dialog.exec():
            # Keep the View-menu Beginner toggle in sync; re-render if it changed.
            if explanations.is_beginner_mode() != was_beginner:
                self.act_beginner.setChecked(explanations.is_beginner_mode())
            # New API keys may unlock richer news/macro — refresh that tab.
            self.results_view.refresh_news()
            self._status("Preferences saved")

    # ── Beginner mode & onboarding ──
    def _on_beginner_toggled(self, enabled: bool) -> None:
        explanations.set_beginner_mode(enabled)
        self.results_view.retheme()  # re-render so blurbs appear/disappear
        state = "on" if enabled else "off"
        self._status(f"Beginner explanations {state}")

    def maybe_show_onboarding(self) -> None:
        """First-run pop-up offering Beginner mode. Shown once."""
        if explanations.is_onboarded():
            return
        explanations.set_onboarded()
        box = QMessageBox(self)
        box.setWindowTitle(f"Welcome to {__app_name__}")
        box.setIcon(QMessageBox.Question)
        box.setText("Turn on Beginner mode?")
        box.setInformativeText(
            "Beginner mode adds plain-English explanations under each result — what it is, "
            "how to read it, and why it matters.\n\nYou can toggle it anytime in "
            "View → Beginner mode. Detailed explanations are always available via the ⓘ icons."
        )
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.button(QMessageBox.Yes).setText("Yes, explain everything")
        box.button(QMessageBox.No).setText("No thanks")
        box.setDefaultButton(QMessageBox.Yes)
        if box.exec() == QMessageBox.Yes:
            self.act_beginner.setChecked(True)  # triggers _on_beginner_toggled

    # ── Results rendering ──
    def display_results(self, results) -> None:
        """Update the headline metrics and populate the result tabs."""
        self._last_results = results
        self._update_metrics(results)
        self.results_view.display(results)

    def _update_metrics(self, results) -> None:
        strip = self.metric_strip
        active = results.active
        passive = results.passive

        active_ret = active.ann_return if active else None
        passive_ret = passive.ann_return if passive else None
        excess = (
            active_ret - passive_ret
            if (active_ret is not None and passive_ret is not None)
            else None
        )
        strip.set_metric("Total Return", fmt_pct(active_ret), delta_str(excess))
        strip.set_metric(
            "Sharpe Ratio",
            f"{active.sharpe:.2f}" if active and not np.isnan(active.sharpe) else "—",
        )
        strip.set_metric("Max Drawdown", fmt_pct(active.max_dd if active else None))
        strip.set_metric("Volatility", fmt_pct(active.ann_vol if active else None))

        alpha_val = beta_val = None
        if results.capm_results:
            alpha_val = float(np.mean([r.alpha for r in results.capm_results]))
            beta_val = float(np.mean([r.beta for r in results.capm_results]))
        strip.set_metric("Alpha", f"{alpha_val * 12:.2%}" if alpha_val is not None else "—")
        strip.set_metric("Beta", f"{beta_val:.2f}" if beta_val is not None else "—")

    # ── Analysis run (threaded) ──
    def _on_run_requested(self, config) -> None:
        """Validated config arrived from the config panel — run it off-thread."""
        if self._thread is not None:
            return  # a run is already in flight

        self._cancelled = False
        self._set_running(True)
        self.progress.setFormat("Starting analysis…")

        self._thread = QThread(self)
        self._worker = AnalysisWorker(config)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        # Tear down the thread once the worker is done, either way.
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    def _on_progress(self, label: str, frac: float) -> None:
        self.progress.setVisible(True)
        self.progress.setValue(int(frac * 100))
        # The step name is shown inside the full-width bar itself.
        self.progress.setFormat(f"   {label}…   %p%")

    def _on_finished(self, results) -> None:
        if self._cancelled:
            return  # user cancelled; discard the late result
        self.progress.setValue(100)
        self.progress.setFormat("   Analysis complete   %p%")
        self.display_results(results)
        self._set_running(False)
        self._status("Analysis complete")

    def _on_failed(self, message: str) -> None:
        if self._cancelled:
            return
        self._set_running(False)
        self._status("Analysis failed")
        QMessageBox.critical(self, "Analysis error", message)

    def _on_cancel(self) -> None:
        """Best-effort cancel: the pipeline has no cooperative stop, so we detach
        the UI and let the worker finish in the background, discarding its result."""
        if self._thread is None:
            return
        self._cancelled = True
        self._set_running(False)
        self._status("Cancelled")

    def _cleanup_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None

    def _set_running(self, running: bool) -> None:
        self.config_panel.set_enabled_for_run(not running)
        self.act_run.setEnabled(not running)
        self.act_cancel.setEnabled(running)
        # During a run the full-width progress bar replaces the idle status label.
        self.status_label.setVisible(not running)
        self.progress.setVisible(running)
        if running:
            self.progress.setValue(0)

    def closeEvent(self, event) -> None:
        """Wait for an in-flight analysis thread so it is not destroyed mid-run."""
        if self._thread is not None and self._thread.isRunning():
            self._cancelled = True
            self._thread.quit()
            self._thread.wait(5000)
        super().closeEvent(event)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {__app_name__}",
            f"<b>{__app_name__}</b> v{__version__}<br><br>"
            "Portfolio optimization and analytics — native desktop edition.",
        )
