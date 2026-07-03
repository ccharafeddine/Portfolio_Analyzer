"""API Keys dialog — optional keys for enhanced data and real-time quotes.

Keys are stored locally via ``AppSettings`` (QSettings) and never leave the machine
except in HTTPS requests to their own provider. All keys are optional: the app works
without them (yfinance data needs none), and each unlocks a richer source. Setting any
one real-time provider key upgrades Live Market Watch from delayed to real-time.
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

# Grouped as (section title, [(settings key, label, help text with a free-key link)]).
_SECTIONS = [
    (
        "Enhanced data",
        [
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
            (
                "FMP_API_KEY",
                "FMP key",
                "Financial Modeling Prep — adds a DCF fair-value estimate on the "
                "Fundamentals tab (yfinance fundamentals work without it). "
                "<a href='https://site.financialmodelingprep.com/developer/docs'>Get a key</a>",
            ),
        ],
    ),
    (
        "Real-time quotes · Live Market Watch",
        [
            (
                "FINNHUB_API_KEY",
                "Finnhub key",
                "Free real-time US quotes — recommended. Upgrades the ticker strip and "
                "watch table from delayed to live. "
                "<a href='https://finnhub.io/register'>Get a key</a>",
            ),
            (
                "POLYGON_API_KEY",
                "Polygon.io key",
                "Real-time / last-trade quotes. "
                "<a href='https://polygon.io/dashboard/signup'>Get a key</a>",
            ),
            (
                "ALPACA_API_KEY",
                "Alpaca key ID",
                "Real-time IEX quotes — needs both the key ID and secret below. "
                "<a href='https://alpaca.markets/'>Get a key</a>",
            ),
            (
                "ALPACA_API_SECRET",
                "Alpaca secret",
                "The secret that pairs with the Alpaca key ID above.",
            ),
        ],
    ),
]


class SettingsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("API Keys")
        self.setMinimumWidth(480)
        self._settings = settings.AppSettings()
        self._edits: dict[str, QLineEdit] = {}

        root = QVBoxLayout(self)

        intro = QLabel(
            "Optional API keys unlock richer data. Set any one real-time provider key to "
            "upgrade Live Market Watch from delayed to live quotes. All keys are stored only "
            "on this computer and are never included in exported reports."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        for title, keys in _SECTIONS:
            header = QLabel(title.upper())
            header.setStyleSheet(
                "color:#94A3B8;font-size:11px;font-weight:700;"
                "letter-spacing:0.06em;margin-top:12px;"
            )
            root.addWidget(header)

            form = QFormLayout()
            form.setSpacing(10)
            for name, label, help_text in keys:
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
