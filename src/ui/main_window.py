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
    QPushButton,
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
from .ticker_strip import TickerStrip
from .widgets.metric_card import MetricStrip
from .worker import AnalysisWorker, QuotesWorker


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
        self._upd_thread = None
        self._upd_worker = None
        self._upd_silent = True
        self._rep_thread = None
        self._rep_worker = None
        self._report_timer = None

        # Seed sample portfolios on first launch so a new user has something to explore.
        try:
            from .samples import seed_sample_portfolios

            seed_sample_portfolios(self._settings)
        except Exception:
            pass

        self._build_menubar()
        self._build_central()
        self._build_statusbar()
        self._init_quotes()

        # Warm up chart views (and the QtWebEngine subsystem) right after the
        # window is shown, so the first Run doesn't flash while creating them.
        QTimer.singleShot(0, self.results_view.prewarm)

        # Auto-check for updates on startup (installed builds only), silent unless a
        # newer version exists. Skipped when running from source during development.
        import sys as _sys

        if getattr(_sys, "frozen", False) and self._settings.value(
            "check_updates_on_startup", True, type=bool
        ):
            QTimer.singleShot(1800, lambda: self._start_update_check(silent=True))

        # Configure the scheduled-reports timer (and run once if set to "On app launch").
        QTimer.singleShot(2500, self._configure_report_schedule)

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
        # Ready-made sample portfolios for quick exploration.
        from .samples import SAMPLE_SPECS

        sample_menu = file_menu.addMenu("Open Sample")
        for name in SAMPLE_SPECS:
            act = QAction(name, self)
            act.triggered.connect(lambda _=False, n=name: self._on_open_sample(n))
            sample_menu.addAction(act)
        file_menu.addSeparator()
        file_menu.addAction(self.act_import_csv)
        file_menu.addAction(self.act_export)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        run_menu = bar.addMenu("&Run")
        self.act_run = QAction("Run Analysis", self, shortcut="Ctrl+R")
        run_menu.addAction(self.act_run)
        run_menu.addSeparator()
        # Entry into the multi-portfolio comparison section, under Run Analysis.
        # (Cancel lives on the progress bar in the status bar during a run.)
        self.act_compare = QAction("Compare Portfolios", self)
        self.act_compare.triggered.connect(self._show_compare_mode)
        run_menu.addAction(self.act_compare)
        # Live Market Watch — delayed quotes for the loaded portfolio.
        self.act_live = QAction("Live Market Watch", self)
        self.act_live.triggered.connect(self._show_live_mode)
        run_menu.addAction(self.act_live)

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
        self.act_updates = QAction("Check for Updates…", self)
        self.act_updates.triggered.connect(self._on_check_updates)
        settings_menu.addAction(self.act_updates)
        settings_menu.addSeparator()
        self.act_sched_reports = QAction("Scheduled Reports…", self)
        self.act_sched_reports.triggered.connect(self._show_scheduled_reports)
        settings_menu.addAction(self.act_sched_reports)
        self.act_settings = QAction("Preferences…", self)
        self.act_settings.triggered.connect(self._show_settings)
        settings_menu.addAction(self.act_settings)

        help_menu = bar.addMenu("&Help")
        self.act_about = QAction("About", self)
        self.act_about.triggered.connect(self._show_about)
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
        from .live_watch_view import LiveWatchView

        self._stack = QStackedWidget()
        self._stack.addWidget(analyze)                 # page 0: Analyze
        self.comparison_view = ComparisonView(
            get_current_config=self.config_panel.build_config
        )
        self.comparison_view.backRequested.connect(self._show_analyze_mode)
        self._stack.addWidget(self.comparison_view)    # page 1: Compare

        self.live_watch_view = LiveWatchView(
            get_current_config=self.config_panel.build_config
        )
        self.live_watch_view.backRequested.connect(self._show_analyze_mode)
        self.live_watch_view.refreshRequested.connect(self._poll_quotes)
        self.live_watch_view.refreshIntervalChanged.connect(self._set_quotes_interval)
        self._stack.addWidget(self.live_watch_view)    # page 2: Live Market Watch
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
        sb.setContentsMargins(0, 0, 0, 0)
        # The always-on ticker strip spans the whole bottom when idle; a
        # full-width progress bar takes over only during a run (the two are
        # mutually exclusive — see ``_set_running``). Transient status messages
        # go through the status bar's own auto-clearing ``showMessage`` so there
        # is no permanent indicator eating into the strip.
        self.ticker_strip = TickerStrip()
        self.ticker_strip.symbolClicked.connect(self._on_ticker_clicked)
        sb.addWidget(self.ticker_strip, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setMinimumHeight(20)
        self.progress.setVisible(False)
        sb.addWidget(self.progress, 1)

        # Cancel sits at the right end of the progress bar, visible only during a run
        # so it's exactly where the user is already looking if it's taking too long.
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("cancelButton")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.setStyleSheet(
            "QPushButton{background:#7f1d1d;color:#fff;border:1px solid #b91c1c;"
            "border-radius:6px;padding:2px 16px;font-weight:600;}"
            "QPushButton:hover{background:#991b1b;}"
        )
        self.btn_cancel.clicked.connect(self._on_cancel)
        sb.addPermanentWidget(self.btn_cancel)

    def _status(self, text: str) -> None:
        # Transient, auto-clearing message — briefly overlays the ticker strip,
        # then the strip returns. No permanent status widget in the bar.
        self.statusBar().showMessage(text, 3500)

    # ── Theming ──
    def _apply_chart_palette(self) -> None:
        from src.charts import plotly_charts

        pal = theme.chart_palette()
        plotly_charts.apply_palette(
            bg=pal["bg"],
            paper=pal["paper"],
            grid=pal["grid"],
            text=pal["text"],
            muted=pal["muted"],
            card=pal["card"],
            border=pal["border"],
            series=pal["series"],
            light=pal["is_light"],
        )

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
        if hasattr(self, "live_watch_view"):
            self.live_watch_view.retheme()
        if hasattr(self, "ticker_strip"):
            self.ticker_strip.retheme()

    # ── Mode switch (Analyze <-> Compare) ──
    def _show_compare_mode(self) -> None:
        self.comparison_view.refresh_portfolio_list()
        self._stack.setCurrentWidget(self.comparison_view)
        self._status("Compare portfolios — pick from the list and click Compare")

    def _show_analyze_mode(self) -> None:
        self._stack.setCurrentIndex(0)

    # ── Live Market Watch ──
    def _show_live_mode(self) -> None:
        self._refresh_watch_universe()
        self._stack.setCurrentWidget(self.live_watch_view)
        self._status("Live Market Watch — delayed quotes")
        self._poll_quotes()  # refresh immediately on entry

    def _on_ticker_clicked(self, symbol: str) -> None:
        self._show_live_mode()

    def _refresh_watch_universe(self):
        """Snapshot the current portfolio config for the strip + live view.
        Keeps the last valid config if the panel is mid-edit / invalid."""
        try:
            cfg = self.config_panel.build_config()
        except Exception:
            cfg = None
        if cfg is not None:
            self._watch_config = cfg
            self.live_watch_view.set_portfolio(cfg)
        return self._watch_config

    def _init_quotes(self) -> None:
        """Wire up delayed-quote polling shared by the strip and the live view."""
        self._watch_config = None
        self._quotes_thread = None
        self._quotes_worker = None
        self._quotes_in_flight = False
        self._quotes_timer = QTimer(self)
        self._quotes_timer.timeout.connect(self._poll_quotes)

        from .live_watch_view import DEFAULT_INTERVAL

        secs = int(self._settings.value("live_refresh_secs", DEFAULT_INTERVAL, type=int))
        self.live_watch_view.set_interval(secs)
        self._set_quotes_interval(secs)
        # First snapshot shortly after launch so the strip comes alive on its own.
        QTimer.singleShot(1200, self._poll_quotes)

    def _set_quotes_interval(self, secs: int) -> None:
        self._settings.setValue("live_refresh_secs", int(secs))
        if secs and secs > 0:
            self._quotes_timer.setInterval(int(secs) * 1000)
            if not self._quotes_timer.isActive():
                self._quotes_timer.start()
        else:
            self._quotes_timer.stop()

    def _poll_quotes(self) -> None:
        if self._quotes_in_flight:
            return
        cfg = self._refresh_watch_universe()
        tickers = list(getattr(cfg, "tickers", []) or []) if cfg else []
        if not tickers:
            return
        self._quotes_in_flight = True
        self._quotes_thread = QThread(self)
        self._quotes_worker = QuotesWorker(tickers)
        self._quotes_worker.moveToThread(self._quotes_thread)
        self._quotes_thread.started.connect(self._quotes_worker.run)
        self._quotes_worker.done.connect(self._on_quotes)
        self._quotes_worker.failed.connect(self._on_quotes_failed)
        self._quotes_worker.done.connect(self._quotes_thread.quit)
        self._quotes_worker.failed.connect(self._quotes_thread.quit)
        self._quotes_thread.finished.connect(self._cleanup_quotes_thread)
        self._quotes_thread.start()

    def _on_quotes(self, quotes: dict) -> None:
        cfg = self._watch_config
        order = list(getattr(cfg, "tickers", []) or []) if cfg else list(quotes.keys())
        self.ticker_strip.set_quotes(quotes, order=order)
        self.live_watch_view.set_quotes(quotes)

    def _on_quotes_failed(self, message: str) -> None:
        pass  # delayed data is best-effort; a failed poll just retries next tick

    def _cleanup_quotes_thread(self) -> None:
        if self._quotes_worker is not None:
            self._quotes_worker.deleteLater()
        if self._quotes_thread is not None:
            self._quotes_thread.deleteLater()
        self._quotes_worker = None
        self._quotes_thread = None
        self._quotes_in_flight = False

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

    def _on_open_sample(self, name: str) -> None:
        from .samples import build_sample_config

        try:
            self.config_panel.load_config(build_sample_config(name))
        except Exception as e:
            QMessageBox.warning(self, "Open Sample", f"Could not load sample:\n{e}")
            return
        self._current_portfolio_path = None
        self._status(f"Loaded sample: {name} — click Run Analysis")

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

    # ── Auto-update ──
    def _on_check_updates(self) -> None:
        self._start_update_check(silent=False)

    def _start_update_check(self, silent: bool) -> None:
        if getattr(self, "_upd_thread", None) is not None:
            return  # a check is already in flight
        from .updater import UpdateCheckWorker

        self._upd_silent = silent
        if not silent:
            self._status("Checking for updates…")
        self._upd_thread = QThread(self)
        self._upd_worker = UpdateCheckWorker()
        self._upd_worker.moveToThread(self._upd_thread)
        self._upd_thread.started.connect(self._upd_worker.run)
        self._upd_worker.done.connect(self._handle_update_result)
        self._upd_worker.failed.connect(self._handle_update_failed)
        self._upd_worker.done.connect(self._upd_thread.quit)
        self._upd_worker.failed.connect(self._upd_thread.quit)
        self._upd_thread.finished.connect(self._cleanup_update_thread)
        self._upd_thread.start()

    def _cleanup_update_thread(self) -> None:
        if getattr(self, "_upd_worker", None) is not None:
            self._upd_worker.deleteLater()
        if getattr(self, "_upd_thread", None) is not None:
            self._upd_thread.deleteLater()
        self._upd_worker = None
        self._upd_thread = None

    def _handle_update_result(self, release) -> None:
        from .updater import is_newer

        latest = release.get("tag", "")
        if is_newer(latest, __version__):
            if not self._upd_silent:
                self._status(f"Update available: {release.get('name') or latest}")
            self._prompt_update(release)
        elif not self._upd_silent:
            self._status("You're on the latest version")
            QMessageBox.information(
                self, "Up to date",
                f"You're running the latest version (v{__version__}).",
            )

    def _handle_update_failed(self, message: str) -> None:
        if not self._upd_silent:
            self._status("Update check failed")
            QMessageBox.warning(
                self, "Update check failed",
                f"Couldn't check for updates right now.\n\n{message}",
            )

    def _prompt_update(self, release) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        notes = (release.get("notes") or "").strip()
        if len(notes) > 700:
            notes = notes[:700].rstrip() + "…"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Update available")
        box.setTextFormat(Qt.RichText)
        box.setText(
            f"<b>{release.get('name') or release.get('tag')}</b> is available."
            f"<br>You're running v{__version__}."
        )
        if notes:
            box.setInformativeText(notes)
        download_btn = box.addButton("Download", QMessageBox.AcceptRole)
        box.addButton("Later", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is download_btn:
            url = release.get("download_url") or release.get("url")
            QDesktopServices.openUrl(QUrl(url))

    # ── Scheduled / automated reports ──
    def _show_scheduled_reports(self) -> None:
        from .scheduled_reports_dialog import ScheduledReportsDialog

        dlg = ScheduledReportsDialog(self)
        dlg.generateNowRequested.connect(self._run_reports)
        dlg.exec()
        self._configure_report_schedule()

    def _configure_report_schedule(self) -> None:
        """Start/stop the in-app report timer from saved settings."""
        s = self._settings
        enabled = s.value("report_sched_enabled", False, type=bool)
        interval = s.value("report_sched_interval", "Daily", type=str)
        if self._report_timer is not None:
            self._report_timer.stop()
            self._report_timer = None
        if not enabled:
            return
        if interval == "On app launch":
            self._run_reports()  # once, now
            return
        ms = {"Hourly": 3_600_000, "Every 6 hours": 21_600_000,
              "Daily": 86_400_000}.get(interval, 86_400_000)
        self._report_timer = QTimer(self)
        self._report_timer.setInterval(ms)
        self._report_timer.timeout.connect(self._run_reports)
        self._report_timer.start()

    def _run_reports(self, out_dir: str = "", formats=None) -> None:
        if getattr(self, "_rep_thread", None) is not None:
            return  # a batch is already running
        from . import paths

        s = self._settings
        out_dir = out_dir or s.value(
            "report_sched_outdir", str(paths.documents_export_dir()), type=str
        )
        if not formats:
            formats = s.value("report_sched_formats", "pdf", type=str).split(",")
        try:
            files = sorted(paths.portfolios_dir().glob("*.json"))
        except Exception:
            files = []
        if not files:
            self._status("Scheduled reports: no saved portfolios to generate")
            return

        from src.config.models import PortfolioConfig

        named = []
        for f in files:
            try:
                named.append((f.stem, PortfolioConfig.load(str(f))))
            except Exception:
                pass
        if not named:
            return

        from .worker import ReportGenWorker

        self._status(f"Generating {len(named)} report(s) in the background…")
        self._rep_thread = QThread(self)
        self._rep_worker = ReportGenWorker(named, out_dir, formats)
        self._rep_worker.moveToThread(self._rep_thread)
        self._rep_thread.started.connect(self._rep_worker.run)
        self._rep_worker.progress.connect(lambda label, _f: self._status(label))
        self._rep_worker.done.connect(self._on_reports_done)
        self._rep_worker.failed.connect(self._on_reports_failed)
        self._rep_worker.done.connect(self._rep_thread.quit)
        self._rep_worker.failed.connect(self._rep_thread.quit)
        self._rep_thread.finished.connect(self._cleanup_report_thread)
        self._rep_thread.start()

    def _cleanup_report_thread(self) -> None:
        if getattr(self, "_rep_worker", None) is not None:
            self._rep_worker.deleteLater()
        if getattr(self, "_rep_thread", None) is not None:
            self._rep_thread.deleteLater()
        self._rep_worker = None
        self._rep_thread = None

    def _on_reports_done(self, written) -> None:
        n = len(written or [])
        self._status(f"Scheduled reports: wrote {n} file(s)")

    def _on_reports_failed(self, message: str) -> None:
        self._status(f"Scheduled reports failed: {message}")

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
        # During a run the full-width progress bar (with its Cancel button) replaces
        # the idle status label + ticker strip; the strip returns when the run ends.
        self.btn_cancel.setVisible(running)
        self.ticker_strip.setVisible(not running)
        self.progress.setVisible(running)
        if running:
            self.progress.setValue(0)

    def closeEvent(self, event) -> None:
        """Wait for an in-flight analysis thread so it is not destroyed mid-run."""
        if getattr(self, "_quotes_timer", None) is not None:
            self._quotes_timer.stop()
        if self._thread is not None and self._thread.isRunning():
            self._cancelled = True
            self._thread.quit()
            self._thread.wait(5000)
        qt = getattr(self, "_quotes_thread", None)
        if qt is not None and qt.isRunning():
            qt.quit()
            qt.wait(3000)
        super().closeEvent(event)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {__app_name__}",
            f"<b>{__app_name__}</b> v{__version__}<br><br>"
            "Portfolio optimization and analytics — native desktop edition.",
        )
