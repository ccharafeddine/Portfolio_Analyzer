"""A WebTab with a thin top toolbar (Refresh + status) and an async-fetch helper.

Base for the News and Macro tabs: both render an HTML page, open links in the
system browser, and fetch their data on a background thread (on every run and on
demand). Subclasses implement ``refresh()`` using ``_start(worker, on_done)``.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QThread, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from .. import theme
from .web_tab import WebTab


class _ExternalLinkPage(QWebEnginePage):
    """Opens clicked http(s) links in the system browser instead of navigating the
    in-app view (which would replace the tab's rendered page)."""

    def acceptNavigationRequest(self, url: QUrl, nav_type, is_main_frame) -> bool:
        if nav_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked:
            if url.scheme() in ("http", "https"):
                QDesktopServices.openUrl(url)
            return False
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)


class RefreshableWebTab(WebTab):
    def __init__(self) -> None:
        super().__init__()

        # Route link clicks to the external browser.
        self._page = _ExternalLinkPage(self._view)
        self._page.setBackgroundColor(self._view.page().backgroundColor())
        self._view.setPage(self._page)

        # Thin toolbar above the single web view.
        bar = QWidget()
        bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        hb = QHBoxLayout(bar)
        hb.setContentsMargins(10, 6, 10, 6)
        hb.setSpacing(10)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setCursor(Qt.PointingHandCursor)
        self._refresh_btn.clicked.connect(self.refresh)
        self._status = QLabel("")
        hb.addWidget(self._refresh_btn)
        hb.addStretch(1)
        hb.addWidget(self._status)

        lay = self.layout()
        lay.insertWidget(0, bar)
        # The toolbar keeps its natural height; the web view takes ALL remaining
        # space (without this, Qt splits the extra height between them).
        lay.setStretch(0, 0)
        lay.setStretch(1, 1)

        self._tickers: list[str] = []
        self._fetching = False
        self._thread: Optional[QThread] = None
        self._worker = None
        self._pending_on_done = None

    # ── ResultsView interface ──
    def set_results(self, results) -> None:
        super().set_results(results)
        try:
            self._tickers = list(results.config.tickers) if results is not None else []
        except Exception:
            self._tickers = []
        self.refresh()

    def refresh(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    # ── Threading helper ──
    def _start(self, worker, on_done) -> None:
        self._fetching = True
        self._refresh_btn.setEnabled(False)
        self._pending_on_done = on_done
        self._thread = QThread(self)
        self._worker = worker
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        # Connect to BOUND METHODS (not local closures) so Qt uses a queued
        # connection back to the UI thread — touching the web view from the
        # worker thread crashes QtWebEngine.
        worker.done.connect(self._handle_done)
        worker.failed.connect(self._handle_failed)
        worker.done.connect(self._thread.quit)
        worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(worker.deleteLater)
        self._thread.start()

    def _handle_done(self, payload) -> None:
        self._fetching = False
        self._refresh_btn.setEnabled(True)
        cb = self._pending_on_done
        if cb is not None:
            cb(payload)

    def _handle_failed(self, msg) -> None:
        self._fetching = False
        self._refresh_btn.setEnabled(True)
        self._set_status(f"Fetch failed: {msg}")

    def _set_status(self, text: str) -> None:
        t = theme.ACTIVE
        self._status.setStyleSheet(f"color:{t.text_muted};font-size:{t.base_pt - 1}px;")
        self._status.setText(text)
