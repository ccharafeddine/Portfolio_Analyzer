"""Configuration panel bound to ``PortfolioConfig``.

Mirrors the Streamlit sidebar 1:1. The panel never re-implements validation:
it builds a ``PortfolioConfig`` and lets Pydantic raise, surfacing the message
inline. On success it emits ``runRequested(config)``.
"""

from __future__ import annotations

from datetime import date

from PySide6.QtCore import QDate, QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .explanations import config_tooltip
from .widgets.collapsible import CollapsibleSection
from .widgets.info_label import InfoLabel

DEFAULT_TICKERS = "AAPL\nMSFT\nNVDA\nGLD\nTLT"


class _WheelGuard(QObject):
    """Stops mouse-wheel events from changing spinbox/combo values while the
    cursor merely passes over them during scrolling. When the widget isn't
    focused, the wheel is redirected to the scroll area so the panel scrolls
    instead of the setting under the cursor changing."""

    def __init__(self, scroll_area) -> None:
        super().__init__(scroll_area)
        self._sa = scroll_area

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt override)
        if event.type() == QEvent.Wheel and not obj.hasFocus():
            QApplication.sendEvent(self._sa.viewport(), event)
            return True  # consume so the spinbox/combo doesn't change value
        return False


def _help(key: str) -> InfoLabel:
    return InfoLabel(config_tooltip(key))


def _row(*widgets) -> QWidget:
    """Pack widgets into a tight left-aligned row (label + '?' badge, etc.)."""
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(5)
    for wd in widgets:
        h.addWidget(wd)
    h.addStretch(1)
    return w


def _section_label(text: str, key: str | None = None) -> QWidget:
    # Styled via QSS (QLabel#sectionLabel) so it re-themes on stylesheet swap.
    lbl = QLabel(text.upper())
    lbl.setObjectName("sectionLabel")
    if key:
        return _row(lbl, _help(key))
    return lbl


def _field_label(text: str, key: str) -> QWidget:
    """A form-row label with a '?' help badge."""
    return _row(QLabel(text), _help(key))


class ConfigPanel(QScrollArea):
    """Emits ``runRequested(PortfolioConfig)`` when the user clicks Run and the
    config validates."""

    runRequested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        body = QWidget()
        root = QVBoxLayout(body)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(8)

        # ── Universe ──
        root.addWidget(_section_label("Universe", "tickers"))
        self.tickers_edit = QPlainTextEdit(DEFAULT_TICKERS)
        self.tickers_edit.setPlaceholderText("One ticker per line")
        self.tickers_edit.setFixedHeight(120)
        root.addWidget(self.tickers_edit)

        self.equal_weights = QCheckBox("Equal weights")
        self.equal_weights.setChecked(True)
        self.equal_weights.toggled.connect(self._on_equal_toggled)
        root.addWidget(_row(self.equal_weights, _help("weights")))

        # ── Weights (shown only when not equal-weighted) ──
        self.weights_group = QGroupBox("Weights")
        wl = QVBoxLayout(self.weights_group)
        self.weights_form = QFormLayout()
        self.weights_form.setLabelAlignment(Qt.AlignLeft)
        self.weights_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.weights_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        wl.addLayout(self.weights_form)
        sync_row = QHBoxLayout()
        self.sync_btn = QPushButton("↻ Sync")
        self.sync_btn.setObjectName("secondary")
        self.sync_btn.clicked.connect(self._rebuild_weights)
        self.sum_label = QLabel("Sum: 0.0000")
        sync_row.addWidget(self.sync_btn)
        sync_row.addStretch(1)
        sync_row.addWidget(self.sum_label)
        wl.addLayout(sync_row)
        self.weights_group.setVisible(False)
        self._weight_spins: dict[str, QDoubleSpinBox] = {}
        root.addWidget(self.weights_group)

        # ── Date range ──
        root.addWidget(_section_label("Date Range", "dates"))
        date_form = QFormLayout()
        date_form.setLabelAlignment(Qt.AlignLeft)
        date_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        date_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.start_date = QDateEdit(QDate(2020, 1, 1))
        self.end_date = QDateEdit(QDate(2025, 12, 31))
        for de in (self.start_date, self.end_date):
            de.setCalendarPopup(True)
            de.setDisplayFormat("yyyy-MM-dd")
        date_form.addRow("Start", self.start_date)
        date_form.addRow("End", self.end_date)
        root.addLayout(date_form)

        # ── Settings ──
        root.addWidget(_section_label("Settings"))
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.benchmark = QLineEdit("SPY")
        form.addRow(_field_label("Benchmark", "benchmark"), self.benchmark)

        self.capital = QDoubleSpinBox()
        self.capital.setRange(1_000, 1_000_000_000)
        self.capital.setDecimals(0)
        self.capital.setSingleStep(100_000)
        self.capital.setGroupSeparatorShown(True)
        self.capital.setValue(1_000_000)
        self.capital.setPrefix("$ ")
        form.addRow(_field_label("Capital", "capital"), self.capital)

        self.rf_rate = QDoubleSpinBox()
        self.rf_rate.setRange(0.0, 0.25)
        self.rf_rate.setDecimals(4)
        self.rf_rate.setSingleStep(0.005)
        self.rf_rate.setValue(0.04)
        self._rf_label = QLabel("Risk-free rate")
        self._rf_from_fred = False
        self.rf_rate.valueChanged.connect(self._on_rf_manual_change)
        form.addRow(_row(self._rf_label, _help("risk_free")), self.rf_rate)
        root.addLayout(form)

        # ── Advanced (tucked away from everyday users) ──
        advanced = CollapsibleSection("Advanced", expanded=False)

        adv_form = QFormLayout()
        adv_form.setLabelAlignment(Qt.AlignLeft)
        adv_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        adv_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.max_bound = QDoubleSpinBox()
        self.max_bound.setRange(0.1, 3.0)
        self.max_bound.setDecimals(2)
        self.max_bound.setSingleStep(0.1)
        self.max_bound.setValue(1.0)
        adv_form.addRow(_field_label("Max weight", "max_weight"), self.max_bound)

        self.y_cp = QDoubleSpinBox()
        self.y_cp.setRange(0.0, 1.0)
        self.y_cp.setDecimals(2)
        self.y_cp.setSingleStep(0.05)
        self.y_cp.setValue(0.80)
        adv_form.addRow(_field_label("ORP fraction", "orp_fraction"), self.y_cp)

        # ── Backtest engine controls ──
        self.inception_mode = QComboBox()
        self.inception_mode.addItem("Rescale to available", "rescale")
        self.inception_mode.addItem("Hold weight in cash", "cash")
        adv_form.addRow(_field_label("New-asset handling", "inception_mode"), self.inception_mode)

        self.rebalance_freq = QComboBox()
        for label, val in [
            ("Buy & hold", "none"), ("Monthly", "monthly"), ("Quarterly", "quarterly"),
            ("Semi-annual", "semiannual"), ("Annual", "annual"),
        ]:
            self.rebalance_freq.addItem(label, val)
        adv_form.addRow(_field_label("Rebalance", "rebalance_frequency"), self.rebalance_freq)

        self.cost_bps = QDoubleSpinBox()
        self.cost_bps.setRange(0.0, 100.0)
        self.cost_bps.setDecimals(1)
        self.cost_bps.setSingleStep(1.0)
        self.cost_bps.setValue(0.0)
        self.cost_bps.setSuffix(" bps")
        adv_form.addRow(_field_label("Transaction cost", "transaction_cost_bps"), self.cost_bps)
        advanced.add_layout(adv_form)

        self.allow_shorts = QCheckBox("Allow shorts")
        self.include_orp = QCheckBox("Include ORP")
        self.include_orp.setChecked(True)
        self.use_dividends = QCheckBox("Use dividend income")
        advanced.add_widget(_row(self.allow_shorts, _help("allow_shorts")))
        advanced.add_widget(_row(self.include_orp, _help("include_orp")))
        advanced.add_widget(_row(self.use_dividends, _help("dividends")))

        # ── Black-Litterman (schema present, editing minimal for now) ──
        self.bl_group = QGroupBox("Black-Litterman views")
        self.bl_group.setCheckable(True)
        self.bl_group.setChecked(False)
        bl_form = QFormLayout(self.bl_group)
        bl_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        bl_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.bl_tau = QDoubleSpinBox()
        self.bl_tau.setRange(0.001, 1.0)
        self.bl_tau.setDecimals(3)
        self.bl_tau.setSingleStep(0.01)
        self.bl_tau.setValue(0.05)
        bl_form.addRow(_field_label("Tau", "bl_tau"), self.bl_tau)
        advanced.add_widget(self.bl_group)

        root.addWidget(advanced)

        # ── Tax (collapsible) ──
        tax = CollapsibleSection("Tax", expanded=False)
        self.tax_enabled = QCheckBox("Enable tax analysis")
        self.tax_enabled.setChecked(True)
        tax.add_widget(_row(self.tax_enabled, _help("tax_enabled")))

        tax_form = QFormLayout()
        tax_form.setLabelAlignment(Qt.AlignLeft)
        tax_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        tax_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.tax_st = QDoubleSpinBox()
        self.tax_st.setRange(0.0, 0.6); self.tax_st.setDecimals(3)
        self.tax_st.setSingleStep(0.01); self.tax_st.setValue(0.35)
        tax_form.addRow(_field_label("Short-term rate", "tax_short"), self.tax_st)
        self.tax_lt = QDoubleSpinBox()
        self.tax_lt.setRange(0.0, 0.4); self.tax_lt.setDecimals(3)
        self.tax_lt.setSingleStep(0.01); self.tax_lt.setValue(0.15)
        tax_form.addRow(_field_label("Long-term rate", "tax_long"), self.tax_lt)
        self.tax_state = QDoubleSpinBox()
        self.tax_state.setRange(0.0, 0.15); self.tax_state.setDecimals(3)
        self.tax_state.setSingleStep(0.01); self.tax_state.setValue(0.0)
        tax_form.addRow(_field_label("State rate", "tax_state"), self.tax_state)
        tax.add_layout(tax_form)

        tax.add_widget(_row(QLabel("Cost basis (optional)"), _help("cost_basis")))
        self.cost_basis_edit = QPlainTextEdit()
        self.cost_basis_edit.setPlaceholderText("One per line:  AAPL: 150.00")
        self.cost_basis_edit.setFixedHeight(80)
        tax.add_widget(self.cost_basis_edit)
        root.addWidget(tax)

        # ── Planning (collapsible) ──
        plan = CollapsibleSection("Planning", expanded=False)
        self.plan_enabled = QCheckBox("Enable retirement planning")
        self.plan_enabled.setChecked(True)
        plan.add_widget(_row(self.plan_enabled, _help("plan_enabled")))

        plan_form = QFormLayout()
        plan_form.setLabelAlignment(Qt.AlignLeft)
        plan_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        plan_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.plan_horizon = QSpinBox()
        self.plan_horizon.setRange(1, 60); self.plan_horizon.setValue(30)
        self.plan_horizon.setSuffix(" yrs")
        plan_form.addRow(_field_label("Horizon", "plan_horizon"), self.plan_horizon)

        self.plan_contrib = QDoubleSpinBox()
        self.plan_contrib.setRange(0, 2_000_000); self.plan_contrib.setDecimals(0)
        self.plan_contrib.setSingleStep(1_000); self.plan_contrib.setGroupSeparatorShown(True)
        self.plan_contrib.setPrefix("$ ")
        plan_form.addRow(_field_label("Contribution/yr", "plan_contribution"), self.plan_contrib)

        self.plan_withdraw = QDoubleSpinBox()
        self.plan_withdraw.setRange(0, 2_000_000); self.plan_withdraw.setDecimals(0)
        self.plan_withdraw.setSingleStep(1_000); self.plan_withdraw.setGroupSeparatorShown(True)
        self.plan_withdraw.setPrefix("$ ")
        plan_form.addRow(_field_label("Withdrawal/yr", "plan_withdrawal"), self.plan_withdraw)

        self.plan_goal = QDoubleSpinBox()
        self.plan_goal.setRange(0, 100_000_000); self.plan_goal.setDecimals(0)
        self.plan_goal.setSingleStep(100_000); self.plan_goal.setGroupSeparatorShown(True)
        self.plan_goal.setPrefix("$ ")
        plan_form.addRow(_field_label("Goal amount", "plan_goal"), self.plan_goal)

        self.plan_inflation = QDoubleSpinBox()
        self.plan_inflation.setRange(0.0, 0.15); self.plan_inflation.setDecimals(3)
        self.plan_inflation.setSingleStep(0.005); self.plan_inflation.setValue(0.025)
        plan_form.addRow(_field_label("Inflation", "plan_inflation"), self.plan_inflation)

        self.plan_exp_return = QDoubleSpinBox()
        self.plan_exp_return.setRange(0.0, 0.30); self.plan_exp_return.setDecimals(3)
        self.plan_exp_return.setSingleStep(0.005); self.plan_exp_return.setValue(0.07)
        plan_form.addRow(_field_label("Expected return", "plan_expected_return"), self.plan_exp_return)
        plan.add_layout(plan_form)
        root.addWidget(plan)

        # ── Error + Run ──
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(f"color: {theme.ACTIVE.red}; font-size: 12px;")
        self.error_label.setVisible(False)
        root.addWidget(self.error_label)

        self.run_btn = QPushButton("▶  Run Analysis")
        self.run_btn.clicked.connect(self._on_run)
        root.addWidget(self.run_btn)
        root.addStretch(1)

        self.setWidget(body)

        # Wheel over a spinbox/combo should scroll the panel, not change the value.
        self._wheel_guard = _WheelGuard(self)
        for w in self.findChildren(QAbstractSpinBox) + self.findChildren(QComboBox):
            w.setFocusPolicy(Qt.StrongFocus)
            w.installEventFilter(self._wheel_guard)

    # ── Weights handling ──
    def _current_tickers(self) -> list[str]:
        raw = self.tickers_edit.toPlainText()
        return [t.strip().upper() for t in raw.splitlines() if t.strip()]

    def _on_equal_toggled(self, checked: bool) -> None:
        self.weights_group.setVisible(not checked)
        if not checked:
            self._rebuild_weights()

    def _rebuild_weights(self) -> None:
        """Rebuild the weight spinboxes from the current ticker list, preserving
        any values already entered for tickers that still exist."""
        prior = {t: s.value() for t, s in self._weight_spins.items()}
        while self.weights_form.rowCount():
            self.weights_form.removeRow(0)
        self._weight_spins.clear()

        tickers = self._current_tickers()
        default = round(1.0 / len(tickers), 4) if tickers else 0.0
        for t in tickers:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 2.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.01)
            spin.setValue(prior.get(t, default))
            spin.valueChanged.connect(self._update_sum)
            self._weight_spins[t] = spin
            self.weights_form.addRow(t, spin)
        self._update_sum()

    def retheme(self) -> None:
        """Re-apply inline-styled colors after a theme switch (kept out of QSS
        because they are dynamic)."""
        self.error_label.setStyleSheet(f"color: {theme.ACTIVE.red}; font-size: 12px;")
        self._update_sum()

    def _update_sum(self) -> None:
        total = sum(s.value() for s in self._weight_spins.values())
        ok = abs(total - 1.0) <= 0.01
        color = theme.ACTIVE.green if ok else theme.ACTIVE.red
        self.sum_label.setStyleSheet(f"color: {color}; font-size: 12px;")
        self.sum_label.setText(f"Sum: {total:.4f}")

    # ── Build config ──
    def _weights_dict(self, tickers: list[str]) -> dict[str, float]:
        if self.equal_weights.isChecked() or not self._weight_spins:
            w = 1.0 / len(tickers) if tickers else 0.0
            return {t: round(w, 6) for t in tickers}
        return {t: float(s.value()) for t, s in self._weight_spins.items() if t in tickers}

    def _parse_cost_basis(self) -> dict[str, float]:
        """Parse the 'TICKER: price' lines into a dict (ignoring bad lines)."""
        out: dict[str, float] = {}
        for line in self.cost_basis_edit.toPlainText().splitlines():
            if ":" not in line:
                continue
            tk, _, val = line.partition(":")
            tk = tk.strip().upper()
            try:
                v = float(val.strip())
                if tk and v > 0:
                    out[tk] = v
            except ValueError:
                continue
        return out

    def set_risk_free_rate(self, value: float, source: str | None = None) -> None:
        """Set the risk-free rate field (e.g. to the latest T-bill yield from FRED).
        ``source`` labels where the value came from, e.g. 'live 3M T-bill'."""
        try:
            lo, hi = self.rf_rate.minimum(), self.rf_rate.maximum()
            self._rf_from_fred = True
            self.rf_rate.setValue(max(lo, min(hi, float(value))))
        except Exception:
            pass
        finally:
            self._rf_from_fred = False
        self._rf_label.setText(f"Risk-free rate ({source})" if source else "Risk-free rate")

    def _on_rf_manual_change(self, _value) -> None:
        # A manual edit clears the "live" label (the value is no longer FRED-driven).
        if self._rf_from_fred:
            return
        if self._rf_label.text() != "Risk-free rate":
            self._rf_label.setText("Risk-free rate")

    def build_config(self):
        """Construct a validated ``PortfolioConfig`` (raises on invalid input)."""
        from src.config.models import (
            BacktestConfig,
            BLConfig,
            CompletePortfolioConfig,
            PlanConfig,
            PortfolioConfig,
            TaxConfig,
        )

        tickers = self._current_tickers()
        weights = self._weights_dict(tickers)
        return PortfolioConfig(
            tickers=tickers,
            weights=weights,
            benchmark=self.benchmark.text(),
            start_date=self.start_date.date().toPython(),
            end_date=self.end_date.date().toPython(),
            capital=float(self.capital.value()),
            risk_free_rate=float(self.rf_rate.value()),
            short_sales=self.allow_shorts.isChecked(),
            max_weight_bound=float(self.max_bound.value()),
            include_orp=self.include_orp.isChecked(),
            include_complete=True,
            use_dividends=self.use_dividends.isChecked(),
            complete_portfolio=CompletePortfolioConfig(y=float(self.y_cp.value())),
            black_litterman=BLConfig(
                enabled=self.bl_group.isChecked(),
                tau=float(self.bl_tau.value()),
                views=[],
            ),
            backtest=BacktestConfig(
                inception_mode=self.inception_mode.currentData(),
                rebalance_frequency=self.rebalance_freq.currentData(),
                transaction_cost_bps=float(self.cost_bps.value()),
            ),
            cost_basis=self._parse_cost_basis(),
            tax=TaxConfig(
                enabled=self.tax_enabled.isChecked(),
                short_term_rate=float(self.tax_st.value()),
                long_term_rate=float(self.tax_lt.value()),
                state_rate=float(self.tax_state.value()),
            ),
            plan=PlanConfig(
                enabled=self.plan_enabled.isChecked(),
                horizon_years=int(self.plan_horizon.value()),
                annual_contribution=float(self.plan_contrib.value()),
                annual_withdrawal=float(self.plan_withdraw.value()),
                goal_amount=float(self.plan_goal.value()),
                inflation=float(self.plan_inflation.value()),
                expected_return=float(self.plan_exp_return.value()),
            ),
        )

    def _on_run(self) -> None:
        self.error_label.setVisible(False)
        try:
            config = self.build_config()
        except Exception as e:
            self._show_error(e)
            return
        self.runRequested.emit(config)

    def _show_error(self, exc: Exception) -> None:
        msg = self._friendly_error(exc)
        self.error_label.setText(msg)
        self.error_label.setVisible(True)

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        """Turn a Pydantic ValidationError into a compact human message."""
        try:
            from pydantic import ValidationError

            if isinstance(exc, ValidationError):
                lines = []
                for err in exc.errors():
                    loc = ".".join(str(p) for p in err.get("loc", ())) or "config"
                    lines.append(f"• {loc}: {err.get('msg', 'invalid')}")
                return "\n".join(lines)
        except Exception:
            pass
        return f"• {exc}"

    def set_enabled_for_run(self, enabled: bool) -> None:
        """Disable inputs while an analysis is running."""
        self.run_btn.setEnabled(enabled)
        self.run_btn.setText("▶  Run Analysis" if enabled else "Running…")
