"""Data tab — report exports (HTML/PDF), holdings, run config, and CSV/ZIP
data exports.

Transcribed from app.py's ``with tab_data:`` block (lines 986-1110). Streamlit
``download_button`` calls become native ``QFileDialog`` save dialogs.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np

from PySide6.QtCore import QBuffer, QByteArray, QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QWidget,
    QFileDialog,
)

from src.charts import plotly_charts as charts
from src.reports.html_builder import build_html_report
from src.reports.pdf_builder import build_pdf_report

from .. import paths
from .. import theme
from ..assets import mark_path
from .base_tab import BaseTab


class _PptxWorker(QObject):
    """Builds the PowerPoint off the UI thread (kaleido chart export is slow)."""

    done = Signal(str)
    failed = Signal(str)

    def __init__(self, results, path, logo_png):
        super().__init__()
        self._results = results
        self._path = path
        self._logo = logo_png

    def run(self):
        try:
            from src.reports.pptx_builder import build_pptx

            build_pptx(self._results, self._path, self._logo)
            self.done.emit(self._path)
        except Exception as e:
            self.failed.emit(str(e))


class DataTab(BaseTab):
    def _populate(self, results) -> None:
        interp = results.interpretations or {}
        self.add_interpretation(interp.get("executive_summary"))

        # ── Reports ──
        self.add_heading("Reports", explain="reports")
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        html_btn = QPushButton("Export HTML Report")
        pdf_btn = QPushButton("Export PDF Report")
        self._pptx_btn = pptx_btn = QPushButton("Export PowerPoint")
        html_btn.clicked.connect(self._export_html)
        pdf_btn.clicked.connect(self._export_pdf)
        pptx_btn.clicked.connect(self._export_pptx)
        row.addWidget(html_btn)
        row.addWidget(pdf_btn)
        row.addWidget(pptx_btn)
        row.addStretch(1)
        self.add_widget(bar)

        # ── Holdings ──
        if results.holdings is not None:
            self.add_heading("Holdings", explain="holdings")
            self.add_table(results.holdings)

        # ── Run configuration ──
        self.add_heading("Run Configuration", explain="run_config")
        cfg_view = QPlainTextEdit(
            json.dumps(results.config.model_dump(mode="json"), indent=2, default=str)
        )
        cfg_view.setReadOnly(True)
        cfg_view.setStyleSheet(
            f"font-family: {theme.ACTIVE.mono}; font-size: {theme.ACTIVE.base_pt - 1}px;"
        )
        cfg_view.setFixedHeight(220)
        self.add_widget(cfg_view)

        # ── Data exports ──
        out = paths.outputs_dir()
        data_files = sorted(f for f in out.iterdir() if f.is_file()) if out.exists() else []
        if data_files:
            self.add_heading("Data Exports", explain="data_exports")
            exp = QWidget()
            erow = QHBoxLayout(exp)
            erow.setContentsMargins(0, 0, 0, 0)
            erow.setSpacing(12)
            zip_btn = QPushButton("Export All Outputs (ZIP)")
            zip_btn.clicked.connect(lambda: self._export_zip(data_files))
            csv_btn = QPushButton("Export a CSV…")
            csv_btn.setObjectName("secondary")
            csv_btn.clicked.connect(lambda: self._export_csv(data_files))
            erow.addWidget(zip_btn)
            erow.addWidget(csv_btn)
            erow.addStretch(1)
            self.add_widget(exp)

    # ── Report figure set (shared with the headless/scheduled generator) ──
    def _report_figures(self) -> dict:
        from src.reports.generate import report_figures

        return report_figures(self._results)

    # ── Export handlers ──
    def _default_path(self, name: str) -> str:
        return str(paths.documents_export_dir() / name)

    def _export_html(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HTML Report", self._default_path("portfolio_report.html"),
            "HTML files (*.html)",
        )
        if not path:
            return
        try:
            html = build_html_report(self._results, chart_figures=self._report_figures())
            Path(path).write_text(html, encoding="utf-8")
            self._info(f"HTML report saved to\n{path}")
        except Exception as e:
            self._error(f"HTML report failed: {e}")

    def _export_pdf(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", self._default_path("portfolio_report.pdf"),
            "PDF files (*.pdf)",
        )
        if not path:
            return
        try:
            pdf = build_pdf_report(self._results, chart_figures=self._report_figures())
            Path(path).write_bytes(pdf)
            self._info(f"PDF report saved to\n{path}")
        except Exception as e:
            self._error(f"PDF report failed: {e}")

    def _logo_png(self):
        """Render the logo mark to PNG bytes for the deck's title slide."""
        try:
            from PySide6.QtGui import QImage, QPainter
            from PySide6.QtSvg import QSvgRenderer

            renderer = QSvgRenderer(mark_path())
            img = QImage(256, 256, QImage.Format_ARGB32)
            img.fill(Qt.transparent)
            painter = QPainter(img)
            renderer.render(painter)
            painter.end()
            ba = QByteArray()
            buf = QBuffer(ba)
            buf.open(QBuffer.WriteOnly)
            img.save(buf, "PNG")
            buf.close()
            return bytes(ba)
        except Exception:
            return None

    def _export_pptx(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PowerPoint", self._default_path("portfolio_presentation.pptx"),
            "PowerPoint (*.pptx)",
        )
        if not path:
            return
        self._pptx_btn.setEnabled(False)
        self._pptx_btn.setText("Generating…")
        self._pptx_thread = QThread(self)
        self._pptx_worker = _PptxWorker(self._results, path, self._logo_png())
        self._pptx_worker.moveToThread(self._pptx_thread)
        self._pptx_thread.started.connect(self._pptx_worker.run)
        self._pptx_worker.done.connect(self._on_pptx_done)
        self._pptx_worker.failed.connect(self._on_pptx_failed)
        self._pptx_worker.done.connect(self._pptx_thread.quit)
        self._pptx_worker.failed.connect(self._pptx_thread.quit)
        self._pptx_thread.finished.connect(self._pptx_worker.deleteLater)
        self._pptx_thread.start()

    def _reset_pptx_btn(self) -> None:
        self._pptx_btn.setEnabled(True)
        self._pptx_btn.setText("Export PowerPoint")

    def _on_pptx_done(self, path: str) -> None:
        self._reset_pptx_btn()
        self._info(f"PowerPoint saved to\n{path}")

    def _on_pptx_failed(self, message: str) -> None:
        self._reset_pptx_btn()
        self._error(f"PowerPoint failed: {message}")

    def _export_zip(self, files: list[Path]) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Outputs ZIP",
            self._default_path("portfolio_analysis_outputs.zip"), "ZIP files (*.zip)",
        )
        if not path:
            return
        try:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in files:
                    zf.write(f, arcname=f.name)
            Path(path).write_bytes(buf.getvalue())
            self._info(f"Saved {len(files)} files to\n{path}")
        except Exception as e:
            self._error(f"ZIP export failed: {e}")

    def _export_csv(self, files: list[Path]) -> None:
        csvs = [f for f in files if f.suffix == ".csv"]
        src, _ = QFileDialog.getOpenFileName(
            self, "Choose a CSV to export", str(paths.outputs_dir()), "CSV files (*.csv)"
        )
        if not src:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save CSV as", self._default_path(Path(src).name), "CSV files (*.csv)"
        )
        if not dest:
            return
        try:
            Path(dest).write_bytes(Path(src).read_bytes())
            self._info(f"Saved to\n{dest}")
        except Exception as e:
            self._error(f"CSV export failed: {e}")

    def _info(self, msg: str) -> None:
        QMessageBox.information(self, "Export", msg)

    def _error(self, msg: str) -> None:
        QMessageBox.warning(self, "Export", msg)
