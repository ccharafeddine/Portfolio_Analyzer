"""Scheduled / automated report generation settings.

Configures automatic report generation for all saved portfolios. Two levels:
- While the app is running: an interval timer (managed by the main window).
- True OS-level scheduling even when the app is closed: this dialog shows the exact
  command to paste into Windows Task Scheduler / cron (which runs the headless CLI).
Settings persist in QSettings; "Generate Now" runs immediately in the background.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from . import paths
from .settings import APP_NAME, ORG_NAME

INTERVALS = ["On app launch", "Hourly", "Every 6 hours", "Daily"]


def scheduler_command(formats: list[str], out_dir: str) -> str:
    """The command an OS scheduler should run for headless report generation."""
    fmt = ",".join(formats) or "pdf"
    py = sys.executable
    if getattr(sys, "frozen", False):
        return f'"{py}" --generate-report --all --format {fmt} --out "{out_dir}"'
    root = Path(__file__).resolve().parents[2]
    return (f'"{py}" "{root / "main_desktop.py"}" --generate-report --all '
            f'--format {fmt} --out "{out_dir}"')


class ScheduledReportsDialog(QDialog):
    generateNowRequested = Signal(str, object)  # (out_dir, [formats])

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scheduled Reports")
        self.setMinimumWidth(540)
        from PySide6.QtCore import QSettings

        self._settings = QSettings(ORG_NAME, APP_NAME)

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.addWidget(QLabel(
            "Automatically generate reports for <b>all saved portfolios</b>."
        ))

        self.enable = QCheckBox("Generate on a schedule while the app is running")
        root.addWidget(self.enable)

        int_row = QHBoxLayout()
        int_row.addWidget(QLabel("Frequency:"))
        self.interval = QComboBox()
        self.interval.addItems(INTERVALS)
        int_row.addWidget(self.interval, 1)
        root.addLayout(int_row)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Formats:"))
        self.fmt_pdf = QCheckBox("PDF")
        self.fmt_html = QCheckBox("HTML")
        fmt_row.addWidget(self.fmt_pdf)
        fmt_row.addWidget(self.fmt_html)
        fmt_row.addStretch(1)
        root.addLayout(fmt_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder:"))
        self.outdir = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        out_row.addWidget(self.outdir, 1)
        out_row.addWidget(browse)
        root.addLayout(out_row)

        gen = QPushButton("Generate Now")
        gen.clicked.connect(self._generate_now)
        root.addWidget(gen)

        root.addWidget(QLabel(
            "For scheduling even when the app is closed, paste this into Windows Task "
            "Scheduler (or cron):"
        ))
        cmd_row = QHBoxLayout()
        self.cmd = QLineEdit()
        self.cmd.setReadOnly(True)
        copy = QPushButton("Copy")
        copy.clicked.connect(self._copy_cmd)
        cmd_row.addWidget(self.cmd, 1)
        cmd_row.addWidget(copy)
        root.addLayout(cmd_row)

        self.status = QLabel("")
        root.addWidget(self.status)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._load()
        for w in (self.fmt_pdf, self.fmt_html, self.outdir):
            (w.stateChanged if isinstance(w, QCheckBox) else w.textChanged).connect(
                self._refresh_cmd
            )
        self._refresh_cmd()

    # ── persistence ──
    def _load(self) -> None:
        s = self._settings
        self.enable.setChecked(s.value("report_sched_enabled", False, type=bool))
        interval = s.value("report_sched_interval", "Daily", type=str)
        self.interval.setCurrentIndex(max(0, INTERVALS.index(interval))
                                      if interval in INTERVALS else 3)
        fmts = s.value("report_sched_formats", "pdf", type=str).split(",")
        self.fmt_pdf.setChecked("pdf" in fmts)
        self.fmt_html.setChecked("html" in fmts)
        if not (self.fmt_pdf.isChecked() or self.fmt_html.isChecked()):
            self.fmt_pdf.setChecked(True)
        self.outdir.setText(
            s.value("report_sched_outdir", str(paths.documents_export_dir()), type=str)
        )

    def formats(self) -> list[str]:
        out = []
        if self.fmt_pdf.isChecked():
            out.append("pdf")
        if self.fmt_html.isChecked():
            out.append("html")
        return out or ["pdf"]

    def _save_and_accept(self) -> None:
        s = self._settings
        s.setValue("report_sched_enabled", self.enable.isChecked())
        s.setValue("report_sched_interval", self.interval.currentText())
        s.setValue("report_sched_formats", ",".join(self.formats()))
        s.setValue("report_sched_outdir", self.outdir.text().strip())
        self.accept()

    # ── actions ──
    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.outdir.text())
        if d:
            self.outdir.setText(d)

    def _refresh_cmd(self) -> None:
        self.cmd.setText(scheduler_command(self.formats(), self.outdir.text().strip()))

    def _copy_cmd(self) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.cmd.text())
        self.status.setText("Command copied to clipboard.")

    def _generate_now(self) -> None:
        self.status.setText("Generating reports in the background…")
        self.generateNowRequested.emit(self.outdir.text().strip(), self.formats())
