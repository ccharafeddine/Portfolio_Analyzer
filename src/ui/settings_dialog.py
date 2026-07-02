"""Preferences dialog — optional API keys for enhanced data sources.

Keys are stored locally via ``AppSettings`` (QSettings) and never leave the machine
except in HTTPS requests to their own provider. All keys are optional: the app works
without them (yfinance news needs none), and each unlocks a richer source.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from . import explanations, settings

# (settings key, label, help text with a link to the free-key page)
_KEYS = [
    (
        "FRED_API_KEY",
        "FRED API key",
        "Free from the St. Louis Fed — powers the Treasury curve and macro rates. "
        "<a href='https://fredaccount.stlouisfed.org/apikeys'>Get a key</a>",
    ),
    (
        "ALPHAVANTAGE_API_KEY",
        "Alpha Vantage key",
        "Adds more news articles and per-article sentiment. "
        "<a href='https://www.alphavantage.co/support/#api-key'>Get a key</a>",
    ),
]


class SettingsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(460)
        self._settings = settings.AppSettings()
        self._edits: dict[str, QLineEdit] = {}

        root = QVBoxLayout(self)

        intro = QLabel(
            "Optional API keys unlock richer data. All keys are stored only on this "
            "computer and are never included in exported reports."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        form.setSpacing(10)
        for name, label, help_text in _KEYS:
            edit = QLineEdit()
            edit.setEchoMode(QLineEdit.Password)
            edit.setPlaceholderText("not set")
            current = settings.get_api_key(name)
            if current:
                edit.setText(current)
            self._edits[name] = edit
            form.addRow(label, edit)
            hint = QLabel(help_text)
            hint.setWordWrap(True)
            hint.setOpenExternalLinks(True)
            hint.setStyleSheet("color:#94A3B8;font-size:11px;")
            form.addRow("", hint)
        root.addLayout(form)

        self._show_keys = QCheckBox("Show keys")
        self._show_keys.toggled.connect(self._toggle_echo)
        root.addWidget(self._show_keys)

        self._beginner = QCheckBox("Beginner mode (show plain-English explanations)")
        self._beginner.setChecked(explanations.is_beginner_mode())
        root.addWidget(self._beginner)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _toggle_echo(self, show: bool) -> None:
        mode = QLineEdit.Normal if show else QLineEdit.Password
        for edit in self._edits.values():
            edit.setEchoMode(mode)

    def accept(self) -> None:
        for name, edit in self._edits.items():
            self._settings.set_api_key(name, edit.text().strip())
        explanations.set_beginner_mode(self._beginner.isChecked())
        super().accept()

    def beginner_changed(self, previous: bool) -> bool:
        """True if the Beginner-mode setting differs from ``previous``."""
        return explanations.is_beginner_mode() != previous
