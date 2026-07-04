"""Scheduled / automated report settings.

Two independent features:

- **Daily Morning Report** — a light Morning Brief (day change, day P&L, today's
  earnings/ex-div, news) for one portfolio, delivered at a chosen clock time via a
  desktop notification and optionally email (SMTP). Fires while the app is running
  and catches up on the next launch if a day was missed. The SMTP password is kept
  in the OS keychain, never in QSettings.
- **Batch reports** — generate the full analytical report for *all* saved
  portfolios on an interval while the app runs, or via an OS scheduler command.

Settings persist in QSettings; "Generate Now" runs immediately in the background.
"""

from __future__ import annotations

from PySide6.QtCore import QTime, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
)

from src.report_cli import scheduler_command

from . import paths, settings
from .settings import APP_NAME, ORG_NAME

INTERVALS = ["On app launch", "Hourly", "Every 6 hours", "Daily"]

# Common SMTP providers → (host, port, use_ssl, setup hint). Selecting one fills
# the server/port/TLS fields; the user still supplies their address + app password.
_SMTP_PRESETS = {
    "Gmail": (
        "smtp.gmail.com", 587, False,
        "Gmail needs a 16-character App Password (Google Account → Security → "
        "2-Step Verification → App passwords), not your normal password.",
    ),
    "Outlook / Office 365": (
        "smtp.office365.com", 587, False,
        "Use your full Microsoft email address. If sign-in is rejected, create an "
        "app password (or ensure SMTP AUTH is enabled for your account).",
    ),
    "iCloud Mail": (
        "smtp.mail.me.com", 587, False,
        "iCloud requires an app-specific password (appleid.apple.com → Sign-In and "
        "Security → App-Specific Passwords).",
    ),
    "Yahoo Mail": (
        "smtp.mail.yahoo.com", 465, True,
        "Yahoo requires an app password (Account Security → Generate app password).",
    ),
}


class ScheduledReportsDialog(QDialog):
    generateNowRequested = Signal(str, object)  # (out_dir, [formats]) — batch
    morningNowRequested = Signal()              # generate the morning report now

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scheduled Reports")
        self.setMinimumWidth(560)
        from PySide6.QtCore import QSettings

        self._settings = QSettings(ORG_NAME, APP_NAME)

        root = QVBoxLayout(self)
        root.setSpacing(10)

        root.addWidget(self._build_morning_group())
        root.addWidget(self._build_batch_group())

        self.status = QLabel("")
        self.status.setWordWrap(True)
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
        self._sync_email_enabled()

    # ── Morning report section ──
    def _build_morning_group(self) -> QGroupBox:
        g = QGroupBox("Daily Morning Report")
        v = QVBoxLayout(g)
        v.setSpacing(8)

        self.m_enable = QCheckBox("Email/notify me a morning brief every day")
        v.addWidget(self.m_enable)

        form = QFormLayout()
        self.m_time = QTimeEdit()
        self.m_time.setDisplayFormat("HH:mm")
        form.addRow("Time (local):", self.m_time)

        self.m_portfolio = QComboBox()
        self.m_portfolio.addItem("Last opened portfolio", "")
        try:
            for p in sorted(paths.portfolios_dir().glob("*.json")):
                self.m_portfolio.addItem(p.stem, str(p))
        except Exception:
            pass
        form.addRow("Portfolio:", self.m_portfolio)
        v.addLayout(form)

        self.m_attach = QCheckBox("Attach the full analytical report")
        v.addWidget(self.m_attach)

        # ── Email delivery sub-section ──
        self.m_email_enable = QCheckBox("Email it (SMTP)")
        self.m_email_enable.toggled.connect(self._sync_email_enabled)
        v.addWidget(self.m_email_enable)

        self._email_box = QGroupBox()
        self._email_box.setFlat(True)
        ef = QFormLayout(self._email_box)
        ef.setContentsMargins(18, 4, 0, 0)
        self.provider = QComboBox()
        self.provider.addItem("Custom", "Custom")
        for _name in _SMTP_PRESETS:
            self.provider.addItem(_name, _name)
        self.provider.currentIndexChanged.connect(self._apply_provider)
        ef.addRow("Provider:", self.provider)
        self._provider_hint = QLabel()
        self._provider_hint.setObjectName("muted")
        self._provider_hint.setWordWrap(True)
        ef.addRow("", self._provider_hint)
        self.smtp_host = QLineEdit()
        self.smtp_host.setPlaceholderText("smtp.gmail.com")
        ef.addRow("SMTP server:", self.smtp_host)
        port_row = QHBoxLayout()
        self.smtp_port = QSpinBox()
        self.smtp_port.setRange(1, 65535)
        self.smtp_port.setValue(587)
        self.smtp_ssl = QCheckBox("Use SSL (port 465)")
        port_row.addWidget(self.smtp_port)
        port_row.addWidget(self.smtp_ssl)
        port_row.addStretch(1)
        ef.addRow("Port:", port_row)
        # Editing server/port/TLS by hand drops the preset back to "Custom".
        self.smtp_host.textChanged.connect(self._on_smtp_manual_edit)
        self.smtp_port.valueChanged.connect(self._on_smtp_manual_edit)
        self.smtp_ssl.toggled.connect(self._on_smtp_manual_edit)
        self.smtp_user = QLineEdit()
        self.smtp_user.setPlaceholderText("you@gmail.com")
        ef.addRow("Username:", self.smtp_user)
        self.smtp_pw = QLineEdit()
        self.smtp_pw.setEchoMode(QLineEdit.Password)
        ef.addRow("Password / app password:", self.smtp_pw)
        self.smtp_from = QLineEdit()
        self.smtp_from.setPlaceholderText("(optional) defaults to the username")
        ef.addRow("From:", self.smtp_from)
        self.m_to = QLineEdit()
        self.m_to.setPlaceholderText("recipient@example.com, second@example.com")
        ef.addRow("Send to:", self.m_to)
        self._keychain_note = QLabel()
        self._keychain_note.setObjectName("muted")
        self._keychain_note.setWordWrap(True)
        ef.addRow("", self._keychain_note)
        v.addWidget(self._email_box)

        row = QHBoxLayout()
        self.test_btn = QPushButton("Send test email")
        self.test_btn.setObjectName("secondary")
        self.test_btn.clicked.connect(self._send_test)
        self.m_gen_btn = QPushButton("Generate morning report now")
        self.m_gen_btn.setObjectName("secondary")
        self.m_gen_btn.clicked.connect(self._morning_now)
        row.addWidget(self.test_btn)
        row.addWidget(self.m_gen_btn)
        row.addStretch(1)
        v.addLayout(row)
        return g

    # ── Batch (all-portfolios) section ──
    def _build_batch_group(self) -> QGroupBox:
        g = QGroupBox("Batch reports — all saved portfolios")
        root = QVBoxLayout(g)
        root.setSpacing(8)

        self.enable = QCheckBox("Generate on a schedule while the app is running")
        root.addWidget(self.enable)

        form = QFormLayout()
        self.interval = QComboBox()
        self.interval.addItems(INTERVALS)
        form.addRow("Frequency:", self.interval)
        fmt_row = QHBoxLayout()
        self.fmt_pdf = QCheckBox("PDF")
        self.fmt_html = QCheckBox("HTML")
        fmt_row.addWidget(self.fmt_pdf)
        fmt_row.addWidget(self.fmt_html)
        fmt_row.addStretch(1)
        form.addRow("Formats:", fmt_row)
        out_row = QHBoxLayout()
        self.outdir = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        out_row.addWidget(self.outdir, 1)
        out_row.addWidget(browse)
        form.addRow("Output folder:", out_row)
        root.addLayout(form)

        gen = QPushButton("Generate batch now")
        gen.clicked.connect(self._generate_now)
        root.addWidget(gen)

        root.addWidget(QLabel(
            "To run even when the app is closed, paste this into Windows Task "
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
        return g

    # ── persistence ──
    def _load(self) -> None:
        s = self._settings
        # Morning report
        self.m_enable.setChecked(s.value("morning_enabled", False, type=bool))
        t = s.value("morning_time", "07:00", type=str) or "07:00"
        self.m_time.setTime(QTime.fromString(t, "HH:mm") if QTime.fromString(t, "HH:mm").isValid()
                            else QTime(7, 0))
        want = s.value("morning_portfolio", "", type=str)
        idx = self.m_portfolio.findData(want)
        self.m_portfolio.setCurrentIndex(idx if idx >= 0 else 0)
        self.m_attach.setChecked(s.value("morning_attach_full", True, type=bool))
        self.m_email_enable.setChecked(s.value("morning_email_enabled", False, type=bool))
        self.smtp_host.setText(s.value("smtp_host", "", type=str))
        self.smtp_port.setValue(int(s.value("smtp_port", 587, type=int)))
        self.smtp_ssl.setChecked(s.value("smtp_use_ssl", False, type=bool))
        self.smtp_user.setText(s.value("smtp_username", "", type=str))
        self.smtp_from.setText(s.value("smtp_from", "", type=str))
        self.m_to.setText(s.value("morning_email_to", "", type=str))
        self._detect_provider()  # reflect a saved host as its provider preset
        # Password lives in the keychain — never shown; a saved one is kept unless
        # the user types a new one.
        user = self.smtp_user.text().strip()
        if settings.get_email_password(user):
            self.smtp_pw.setPlaceholderText("•••••• saved — leave blank to keep")
        if not settings.keyring_available():
            self._keychain_note.setText(
                "⚠ No OS keychain backend is available, so the email password "
                "can't be stored securely. Email delivery will be unavailable."
            )
        else:
            self._keychain_note.setText(
                "The password is stored in your OS keychain (Windows Credential "
                "Locker / macOS Keychain), not in the app's settings."
            )
        # Batch
        self.enable.setChecked(s.value("report_sched_enabled", False, type=bool))
        interval = s.value("report_sched_interval", "Daily", type=str)
        self.interval.setCurrentIndex(INTERVALS.index(interval) if interval in INTERVALS else 3)
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
        # Morning report
        s.setValue("morning_enabled", self.m_enable.isChecked())
        s.setValue("morning_time", self.m_time.time().toString("HH:mm"))
        s.setValue("morning_portfolio", self.m_portfolio.currentData() or "")
        s.setValue("morning_attach_full", self.m_attach.isChecked())
        s.setValue("morning_email_enabled", self.m_email_enable.isChecked())
        s.setValue("smtp_host", self.smtp_host.text().strip())
        s.setValue("smtp_port", int(self.smtp_port.value()))
        s.setValue("smtp_use_ssl", self.smtp_ssl.isChecked())
        s.setValue("smtp_username", self.smtp_user.text().strip())
        s.setValue("smtp_from", self.smtp_from.text().strip())
        s.setValue("morning_email_to", self.m_to.text().strip())
        s.setValue("morning_outdir", self.outdir.text().strip())
        # Store a newly-typed password in the keychain (blank keeps the existing).
        pw = self.smtp_pw.text()
        user = self.smtp_user.text().strip()
        if pw and user:
            settings.set_email_password(user, pw)
        # Batch
        s.setValue("report_sched_enabled", self.enable.isChecked())
        s.setValue("report_sched_interval", self.interval.currentText())
        s.setValue("report_sched_formats", ",".join(self.formats()))
        s.setValue("report_sched_outdir", self.outdir.text().strip())
        self.accept()

    # ── actions ──
    def _apply_provider(self, *_a) -> None:
        """Fill server/port/TLS from the chosen provider preset (no-op for Custom)."""
        preset = _SMTP_PRESETS.get(self.provider.currentData())
        if preset is None:
            self._provider_hint.setText("")
            return
        host, port, ssl, hint = preset
        for w in (self.smtp_host, self.smtp_port, self.smtp_ssl):
            w.blockSignals(True)
        self.smtp_host.setText(host)
        self.smtp_port.setValue(port)
        self.smtp_ssl.setChecked(ssl)
        for w in (self.smtp_host, self.smtp_port, self.smtp_ssl):
            w.blockSignals(False)
        self._provider_hint.setText(hint)

    def _on_smtp_manual_edit(self, *_a) -> None:
        """When the fields diverge from the selected preset, revert to Custom."""
        preset = _SMTP_PRESETS.get(self.provider.currentData())
        if preset and (self.smtp_host.text().strip() != preset[0]
                       or int(self.smtp_port.value()) != preset[1]
                       or self.smtp_ssl.isChecked() != preset[2]):
            self.provider.blockSignals(True)
            self.provider.setCurrentIndex(0)  # Custom
            self.provider.blockSignals(False)
            self._provider_hint.setText("")

    def _detect_provider(self) -> None:
        """Select the provider whose preset matches the loaded server/port/TLS."""
        host = self.smtp_host.text().strip().lower()
        port = int(self.smtp_port.value())
        ssl = self.smtp_ssl.isChecked()
        for name, (h, p, s, hint) in _SMTP_PRESETS.items():
            if host == h and port == p and ssl == s:
                idx = self.provider.findData(name)
                if idx >= 0:
                    self.provider.blockSignals(True)
                    self.provider.setCurrentIndex(idx)
                    self.provider.blockSignals(False)
                    self._provider_hint.setText(hint)
                return

    def _sync_email_enabled(self, *_a) -> None:
        self._email_box.setEnabled(self.m_email_enable.isChecked())
        self.test_btn.setEnabled(self.m_email_enable.isChecked())

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
        self.status.setText("Generating batch reports in the background…")
        self.generateNowRequested.emit(self.outdir.text().strip(), self.formats())

    def _morning_now(self) -> None:
        self._save_settings_only()
        self.status.setText("Generating the morning report in the background…")
        self.morningNowRequested.emit()

    def _save_settings_only(self) -> None:
        """Persist without closing, so 'now' actions use the current fields."""
        pw = self.smtp_pw.text()
        user = self.smtp_user.text().strip()
        if pw and user:
            settings.set_email_password(user, pw)
        s = self._settings
        s.setValue("morning_portfolio", self.m_portfolio.currentData() or "")
        s.setValue("morning_attach_full", self.m_attach.isChecked())
        s.setValue("morning_email_enabled", self.m_email_enable.isChecked())
        s.setValue("smtp_host", self.smtp_host.text().strip())
        s.setValue("smtp_port", int(self.smtp_port.value()))
        s.setValue("smtp_use_ssl", self.smtp_ssl.isChecked())
        s.setValue("smtp_username", user)
        s.setValue("smtp_from", self.smtp_from.text().strip())
        s.setValue("morning_email_to", self.m_to.text().strip())
        s.setValue("morning_outdir", self.outdir.text().strip())

    def _send_test(self) -> None:
        from PySide6.QtCore import Qt as _Qt
        from PySide6.QtWidgets import QApplication

        from src.reports.emailer import SmtpConfig, send_email

        user = self.smtp_user.text().strip()
        pw = self.smtp_pw.text() or settings.get_email_password(user) or ""
        to = [a.strip() for a in self.m_to.text().split(",") if a.strip()]
        host = self.smtp_host.text().strip()
        if not (host and user and pw and to):
            self.status.setText(
                "Fill SMTP server, username, password, and at least one recipient first."
            )
            return
        cfg = SmtpConfig(
            host=host, port=int(self.smtp_port.value()), username=user, password=pw,
            from_addr=self.smtp_from.text().strip() or user,
            use_ssl=self.smtp_ssl.isChecked(), timeout=20,
        )
        QApplication.setOverrideCursor(_Qt.WaitCursor)
        try:
            send_email(cfg, to, "Portfolio Analyzer — test email",
                       "<p>Your Portfolio Analyzer morning-report email is configured "
                       "correctly. ✅</p>")
            self.status.setText("Test email sent ✓")
        except Exception as e:
            self.status.setText(f"Test failed: {e}")
        finally:
            QApplication.restoreOverrideCursor()
