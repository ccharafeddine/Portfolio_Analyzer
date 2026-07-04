"""Configuration panel bound to ``PortfolioConfig``.

Mirrors the Streamlit sidebar 1:1. The panel never re-implements validation:
it builds a ``PortfolioConfig`` and lets Pydantic raise, surfacing the message
inline. On success it emits ``runRequested(config)``.
"""

from __future__ import annotations

from datetime import date

from PySide6.QtCore import QDate, QEvent, QObject, Qt, QThread, QTimer, Signal
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


def _bl_views_to_text(views) -> str:
    """Serialize BLView objects back to the editor's line syntax."""
    lines = []
    for v in views or []:
        conf = "" if v.confidence == "medium" else f" @{v.confidence}"
        pct = f"{v.q * 100:g}%"
        if v.type == "relative":
            lines.append(f"{v.asset_long} > {v.asset_short}: {pct}{conf}")
        else:
            lines.append(f"{v.asset}: {pct}{conf}")
    return "\n".join(lines)


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
    # Emitted whenever the whole config is replaced (open / sample / new / import),
    # so the live layer can re-sync its ticker universe immediately.
    configChanged = Signal()

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

        # ── Allocation (shown only when not equal-weighted): weights or shares ──
        self.weights_group = QGroupBox("Allocation")
        wl = QVBoxLayout(self.weights_group)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Enter by"))
        self.alloc_mode = QComboBox()
        self.alloc_mode.addItem("Weights", "weights")
        self.alloc_mode.addItem("Shares", "shares")
        self.alloc_mode.setToolTip(
            "Weights: target allocation fractions.\n"
            "Shares: how many shares you hold — Calculate sets Capital and weights "
            "from current prices."
        )
        self.alloc_mode.currentIndexChanged.connect(self._on_alloc_mode_changed)
        mode_row.addWidget(self.alloc_mode)
        mode_row.addStretch(1)
        wl.addLayout(mode_row)

        self.weights_form = QFormLayout()
        self.weights_form.setLabelAlignment(Qt.AlignLeft)
        self.weights_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.weights_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        wl.addLayout(self.weights_form)

        # Cash held alongside the stocks (shares mode only). Kept separate from the
        # invested Capital; total account value = Capital + Cash.
        self.cash_container = QWidget()
        cash_row = QHBoxLayout(self.cash_container)
        cash_row.setContentsMargins(0, 4, 0, 0)
        cash_row.addWidget(QLabel("Cash"))
        self.cash_input = QDoubleSpinBox()
        self.cash_input.setRange(0.0, 1_000_000_000.0)
        self.cash_input.setDecimals(0)
        self.cash_input.setSingleStep(1_000)
        self.cash_input.setGroupSeparatorShown(True)
        self.cash_input.setPrefix("$ ")
        self.cash_input.valueChanged.connect(self._update_sum)
        cash_row.addWidget(self.cash_input, 1)
        self.cash_container.setVisible(False)
        wl.addWidget(self.cash_container)

        sync_row = QHBoxLayout()
        self.sync_btn = QPushButton("Calculate weights")
        self.sync_btn.setObjectName("secondary")
        self.sync_btn.setCursor(Qt.PointingHandCursor)
        self.sync_btn.clicked.connect(self._on_alloc_button)
        self.sum_label = QLabel("Sum: 0.0000")
        self.sum_label.setWordWrap(True)
        sync_row.addWidget(self.sync_btn)
        sync_row.addSpacing(10)
        sync_row.addWidget(self.sum_label, 1)
        wl.addLayout(sync_row)
        self.weights_group.setVisible(False)
        self._weight_spins: dict[str, QDoubleSpinBox] = {}
        self._derived_weights: dict[str, float] = {}  # value-based, from shares×price
        self._calc_thread: QThread | None = None
        self._calc_worker = None
        self._calc_shares: dict[str, float] = {}
        root.addWidget(self.weights_group)

        # Keep the allocation rows in sync with the ticker list automatically
        # (debounced), so there's no manual "sync" step in weights mode.
        self._row_sync_timer = QTimer(self)
        self._row_sync_timer.setSingleShot(True)
        self._row_sync_timer.setInterval(500)
        self._row_sync_timer.timeout.connect(self._auto_sync_rows)
        self.tickers_edit.textChanged.connect(self._row_sync_timer.start)

        # ── Date range ──
        root.addWidget(_section_label("Date Range", "dates"))
        date_form = QFormLayout()
        date_form.setLabelAlignment(Qt.AlignLeft)
        date_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        date_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        # Default range = the trailing 5 years (today − 5y → today), so a fresh
        # analysis covers a sensible recent window out of the box.
        self.start_date = QDateEdit(QDate.currentDate().addYears(-5))
        self.end_date = QDateEdit(QDate.currentDate())
        for de in (self.start_date, self.end_date):
            de.setCalendarPopup(True)
            de.setDisplayFormat("yyyy-MM-dd")
        date_form.addRow("Start", self.start_date)
        # End field + a "Today" button that snaps it to the current date (handy after
        # loading an older saved portfolio, whose stored end date may be stale).
        end_row = QHBoxLayout()
        end_row.setSpacing(6)
        end_row.addWidget(self.end_date, 1)
        self.end_today_btn = QPushButton("Today")
        self.end_today_btn.setObjectName("secondary")
        self.end_today_btn.setCursor(Qt.PointingHandCursor)
        self.end_today_btn.setToolTip("Set the end date to today")
        self.end_today_btn.clicked.connect(
            lambda: self.end_date.setDate(QDate.currentDate())
        )
        end_row.addWidget(self.end_today_btn)
        date_form.addRow("End", end_row)
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
        self.capital.valueChanged.connect(self._update_total_value)
        form.addRow(_field_label("Capital", "capital"), self.capital)
        # Total account value = invested Capital + Cash. Shown only when a cash
        # balance is set (shares mode), so the full number is visible even though
        # Capital itself stays the invested amount the analysis uses.
        self.total_value_label = QLabel()
        self.total_value_label.setWordWrap(True)
        self.total_value_label.setVisible(False)
        form.addRow("", self.total_value_label)

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

        # Optional blended benchmark (e.g. 60/40). When filled, the Benchmark field
        # above becomes the label for this mix.
        advanced.add_widget(_field_label("Benchmark blend", "benchmark_blend"))
        self.benchmark_blend = QPlainTextEdit()
        self.benchmark_blend.setPlaceholderText("Optional, one per line:  SPY: 0.6")
        self.benchmark_blend.setFixedHeight(70)
        advanced.add_widget(self.benchmark_blend)

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
        self.bl_views_edit = QPlainTextEdit()
        self.bl_views_edit.setPlaceholderText(
            "One view per line:\nAAPL: 12%           (AAPL returns ~12%/yr)\n"
            "AAPL > MSFT: 3%     (AAPL beats MSFT by 3%)\nadd @high or @low for confidence"
        )
        self.bl_views_edit.setFixedHeight(96)
        bl_form.addRow(_field_label("Views", "bl_views"))
        bl_form.addRow(self.bl_views_edit)
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

        # Snapshot the initial (default) configuration for "New Portfolio".
        self._defaults = self.build_config()

    # ── Weights handling ──
    def _current_tickers(self) -> list[str]:
        raw = self.tickers_edit.toPlainText()
        return [t.strip().upper() for t in raw.splitlines() if t.strip()]

    def _on_equal_toggled(self, checked: bool) -> None:
        self.weights_group.setVisible(not checked)
        if not checked:
            self._rebuild_weights()
            self._update_alloc_ui()

    def _auto_sync_rows(self) -> None:
        """Rebuild the allocation rows to match the current ticker list. Runs
        (debounced) whenever the tickers change, so weights/shares rows stay in
        sync without a manual button."""
        if not self.weights_group.isVisible():
            return
        if self._alloc_is_shares():
            self._derived_weights = {}  # universe changed — force a recalculation
        self._rebuild_weights(preserve=True)

    # ── Allocation mode (weights vs shares) ──
    def _alloc_is_shares(self) -> bool:
        return self.alloc_mode.currentData() == "shares"

    def _set_alloc_mode(self, mode: str) -> None:
        idx = 1 if mode == "shares" else 0
        self.alloc_mode.blockSignals(True)
        self.alloc_mode.setCurrentIndex(idx)
        self.alloc_mode.blockSignals(False)
        self._update_alloc_ui()

    def _on_alloc_mode_changed(self, _idx: int) -> None:
        self._derived_weights = {}
        self._rebuild_weights(preserve=False)  # values mean different things per mode
        self._update_alloc_ui()

    def _update_alloc_ui(self) -> None:
        # The button + cash field only exist in shares mode (weights rows auto-sync).
        shares = self._alloc_is_shares()
        self.sync_btn.setVisible(shares)
        self.cash_container.setVisible(shares)
        self.sync_btn.setText("Calculate weights")
        self.sync_btn.setToolTip(
            "Fetch current prices, set Capital = shares × price, and derive the weights"
        )
        self._update_sum()

    def _on_alloc_button(self) -> None:
        self._calculate_from_shares()

    def _rebuild_weights(self, preserve: bool = True) -> None:
        """Rebuild the allocation spinboxes from the current ticker list. In
        weights mode the rows are target weights; in shares mode they are share
        counts. Values for still-present tickers are preserved within a mode."""
        prior = {t: s.value() for t, s in self._weight_spins.items()} if preserve else {}
        while self.weights_form.rowCount():
            self.weights_form.removeRow(0)
        self._weight_spins.clear()

        tickers = self._current_tickers()
        shares_mode = self._alloc_is_shares()
        default = 0.0 if shares_mode else (round(1.0 / len(tickers), 4) if tickers else 0.0)
        for t in tickers:
            spin = QDoubleSpinBox()
            if shares_mode:
                spin.setRange(0.0, 1_000_000_000.0)
                spin.setDecimals(4)
                spin.setSingleStep(1.0)
                spin.setSuffix(" sh")
                spin.valueChanged.connect(self._on_shares_changed)
            else:
                spin.setRange(0.0, 2.0)
                spin.setDecimals(4)
                spin.setSingleStep(0.01)
                spin.valueChanged.connect(self._update_sum)
            spin.setValue(prior.get(t, default))
            self._weight_spins[t] = spin
            self.weights_form.addRow(t, spin)
        self._update_sum()

    def _on_shares_changed(self) -> None:
        # Editing shares invalidates the last price-based calculation.
        self._derived_weights = {}
        self._update_sum()

    def retheme(self) -> None:
        """Re-apply inline-styled colors after a theme switch (kept out of QSS
        because they are dynamic)."""
        self.error_label.setStyleSheet(f"color: {theme.ACTIVE.red}; font-size: 12px;")
        self._update_sum()

    def _update_total_value(self) -> None:
        """Show 'Total account value = invested + cash' when a cash balance is set."""
        lbl = getattr(self, "total_value_label", None)
        if lbl is None:
            return
        cash = self.cash_input.value() if hasattr(self, "cash_input") else 0.0
        if self._alloc_is_shares() and cash > 0:
            cap = self.capital.value()
            t = theme.ACTIVE
            lbl.setStyleSheet(f"color: {t.text_slate}; font-size: 12px;")
            lbl.setText(
                f"Total account value ${cap + cash:,.0f}  "
                f"(invested ${cap:,.0f} + cash ${cash:,.0f})"
            )
            lbl.setVisible(True)
        else:
            lbl.setVisible(False)

    def _update_sum(self) -> None:
        self._update_total_value()
        t = theme.ACTIVE
        if self._alloc_is_shares():
            if self._derived_weights:
                cap = self.capital.value()
                cash = self.cash_input.value()
                self.sum_label.setStyleSheet(f"color: {t.green}; font-size: 12px;")
                if cash > 0:
                    self.sum_label.setText(
                        f"Capital ${cap:,.0f} + Cash ${cash:,.0f} = ${cap + cash:,.0f}"
                    )
                else:
                    self.sum_label.setText(f"Capital ${cap:,.0f} · weights set")
            else:
                self.sum_label.setStyleSheet(f"color: {t.text_muted}; font-size: 12px;")
                self.sum_label.setText("also sets Capital from prices")
            return
        total = sum(s.value() for s in self._weight_spins.values())
        ok = abs(total - 1.0) <= 0.01
        color = t.green if ok else t.red
        self.sum_label.setStyleSheet(f"color: {color}; font-size: 12px;")
        self.sum_label.setText(f"Sum: {total:.4f}")

    # ── Shares → capital + weights (async price fetch) ──
    def _calculate_from_shares(self) -> None:
        shares = {t: s.value() for t, s in self._weight_spins.items() if s.value() > 0}
        if not shares:
            self.sum_label.setStyleSheet(f"color: {theme.ACTIVE.red}; font-size: 12px;")
            self.sum_label.setText("Enter share counts first")
            return
        if self._calc_thread is not None:
            return  # a calculation is already running
        from .worker import QuotesWorker

        self._calc_shares = shares
        self.sync_btn.setEnabled(False)
        self.sum_label.setStyleSheet(f"color: {theme.ACTIVE.text_muted}; font-size: 12px;")
        self.sum_label.setText("Fetching prices…")

        self._calc_thread = QThread(self)
        self._calc_worker = QuotesWorker(list(shares))
        self._calc_worker.moveToThread(self._calc_thread)
        self._calc_thread.started.connect(self._calc_worker.run)
        self._calc_worker.done.connect(self._on_calc_quotes)
        self._calc_worker.failed.connect(self._on_calc_failed)
        self._calc_worker.done.connect(self._calc_thread.quit)
        self._calc_worker.failed.connect(self._calc_thread.quit)
        self._calc_thread.finished.connect(self._cleanup_calc)
        self._calc_thread.start()

    def _on_calc_quotes(self, quotes: dict) -> None:
        from .allocation import shares_to_weights_and_capital

        prices = {}
        for t in self._calc_shares:
            last = getattr(quotes.get(t), "last", None)
            if isinstance(last, (int, float)) and last > 0:
                prices[t] = last
        weights, capital = shares_to_weights_and_capital(self._calc_shares, prices)
        if not weights:
            self.sum_label.setStyleSheet(f"color: {theme.ACTIVE.red}; font-size: 12px;")
            self.sum_label.setText("No prices available — try again")
            return
        self._derived_weights = weights
        self.capital.setValue(round(capital))
        cash = self.cash_input.value()
        missing = [t for t in self._calc_shares if t not in prices]
        if cash > 0:
            msg = f"Capital ${capital:,.0f} + Cash ${cash:,.0f} = ${capital + cash:,.0f}"
        else:
            msg = f"Capital ${capital:,.0f} · weights set"
        if missing:
            msg += f" · no price: {', '.join(missing)}"
        self.sum_label.setStyleSheet(f"color: {theme.ACTIVE.green}; font-size: 12px;")
        self.sum_label.setText(msg)

    def _on_calc_failed(self, _message: str) -> None:
        self.sum_label.setStyleSheet(f"color: {theme.ACTIVE.red}; font-size: 12px;")
        self.sum_label.setText("Price fetch failed — check your connection")

    def _cleanup_calc(self) -> None:
        if self._calc_worker is not None:
            self._calc_worker.deleteLater()
        if self._calc_thread is not None:
            self._calc_thread.deleteLater()
        self._calc_worker = None
        self._calc_thread = None
        self.sync_btn.setEnabled(True)

    # ── Build config ──
    def _weights_dict(self, tickers: list[str]) -> dict[str, float]:
        if self.equal_weights.isChecked() or not self._weight_spins:
            w = 1.0 / len(tickers) if tickers else 0.0
            return {t: round(w, 6) for t in tickers}
        if self._alloc_is_shares():
            # Prefer the price-based weights from the last Calculate; otherwise fall
            # back to share-count proportions so a run still validates.
            if self._derived_weights:
                return {t: w for t, w in self._derived_weights.items() if t in tickers}
            shares = {t: s.value() for t, s in self._weight_spins.items()
                      if t in tickers and s.value() > 0}
            total = sum(shares.values())
            if total > 0:
                return {t: v / total for t, v in shares.items()}
            w = 1.0 / len(tickers) if tickers else 0.0
            return {t: round(w, 6) for t in tickers}
        return {t: float(s.value()) for t, s in self._weight_spins.items() if t in tickers}

    def _shares_dict(self) -> dict[str, float]:
        """Share counts when in shares mode (else empty)."""
        if self.equal_weights.isChecked() or not self._alloc_is_shares():
            return {}
        return {t: float(s.value()) for t, s in self._weight_spins.items() if s.value() > 0}

    def set_cost_basis(self, cost_basis: dict) -> None:
        """Replace the cost-basis entry with ``{ticker: price}`` (positive only)."""
        lines = []
        for k, v in (cost_basis or {}).items():
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            if val > 0:
                lines.append(f"{str(k).strip().upper()}: {val}")
        self.cost_basis_edit.setPlainText("\n".join(lines))

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

    def _parse_bl_views(self) -> list:
        """Parse the free-text views box into BLView objects (bad lines ignored).

        Absolute:  ``AAPL: 12%``   Relative:  ``AAPL > MSFT: 3%``
        Optional trailing confidence:  ``AAPL: 12% @high`` (low / medium / high).
        """
        from src.config.models import BLView

        out = []
        for raw in self.bl_views_edit.toPlainText().splitlines():
            line = raw.strip()
            if not line:
                continue
            conf = "medium"
            if "@" in line:
                line, _, ctok = line.rpartition("@")
                line = line.strip()
                ctok = ctok.strip().lower()
                if ctok.startswith("l"):
                    conf = "low"
                elif ctok.startswith("h"):
                    conf = "high"
            if ":" not in line:
                continue
            lhs, _, qs = line.partition(":")
            lhs = lhs.strip()
            try:
                q = float(qs.strip().replace("%", "")) / 100.0
            except ValueError:
                continue
            try:
                if ">" in lhs:
                    a, b = (s.strip().upper() for s in lhs.split(">", 1))
                    if a and b and a != b:
                        out.append(BLView(type="relative", asset_long=a,
                                          asset_short=b, q=q, confidence=conf))
                elif lhs:
                    out.append(BLView(type="absolute", asset=lhs.upper(),
                                      q=q, confidence=conf))
            except Exception:
                continue
        return out

    def _parse_benchmark_blend(self) -> dict[str, float]:
        """Parse 'TICKER: weight' lines into a blend dict (ignoring bad lines).
        Accepts 'SPY: 0.6' or 'SPY 60'; weights are used as-is (engine normalizes)."""
        out: dict[str, float] = {}
        for line in self.benchmark_blend.toPlainText().splitlines():
            line = line.strip()
            if not line:
                continue
            sep = ":" if ":" in line else (" " if " " in line else None)
            if sep is None:
                continue
            tk, _, val = line.partition(sep)
            tk = tk.strip().upper()
            try:
                w = float(val.strip())
                if tk and w > 0:
                    out[tk] = w
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

    # ── Load / reset / import (File menu) ──
    @staticmethod
    def _select_combo_data(combo, value) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def load_config(self, config) -> None:
        """Populate every field from a PortfolioConfig (the reverse of build_config)."""
        c = config
        self.tickers_edit.setPlainText("\n".join(c.tickers))
        self.benchmark.setText(c.benchmark)
        self.start_date.setDate(QDate(c.start_date.year, c.start_date.month, c.start_date.day))
        self.end_date.setDate(QDate(c.end_date.year, c.end_date.month, c.end_date.day))
        self.capital.setValue(float(c.capital))
        self.cash_input.setValue(float(getattr(c, "cash", 0.0) or 0.0))
        self.set_risk_free_rate(float(c.risk_free_rate))

        # Allocation: shares mode if the config carries share counts; else weights
        # (equal when every weight matches 1/N).
        weights = dict(c.weights or {})
        shares = dict(getattr(c, "shares", {}) or {})
        n = len(c.tickers)
        equal = (not shares) and (
            (not weights)
            or (n > 0 and all(abs(weights.get(t, 0.0) - 1.0 / n) <= 1e-4 for t in c.tickers))
        )
        self.equal_weights.setChecked(equal)
        self.weights_group.setVisible(not equal)
        self._derived_weights = {}
        self._set_alloc_mode("shares" if shares else "weights")
        if not equal:
            self._rebuild_weights(preserve=False)
            source = shares if shares else weights
            for t, spin in self._weight_spins.items():
                if t in source:
                    spin.setValue(float(source[t]))
            # The saved weights were derived from these shares × price at save time.
            if shares:
                self._derived_weights = dict(weights)
            self._update_sum()

        # Advanced
        self.allow_shorts.setChecked(bool(c.short_sales))
        self.max_bound.setValue(float(c.max_weight_bound))
        self.include_orp.setChecked(bool(c.include_orp))
        self.use_dividends.setChecked(bool(c.use_dividends))
        self.y_cp.setValue(float(c.complete_portfolio.y))
        self.bl_group.setChecked(bool(c.black_litterman.enabled))
        self.bl_tau.setValue(float(c.black_litterman.tau))
        self.bl_views_edit.setPlainText(_bl_views_to_text(c.black_litterman.views))
        self._select_combo_data(self.inception_mode, c.backtest.inception_mode)
        self._select_combo_data(self.rebalance_freq, c.backtest.rebalance_frequency)
        self.cost_bps.setValue(float(c.backtest.transaction_cost_bps))
        self.cost_basis_edit.setPlainText(
            "\n".join(f"{k}: {v}" for k, v in (c.cost_basis or {}).items())
        )
        self.benchmark_blend.setPlainText(
            "\n".join(f"{k}: {v}" for k, v in (getattr(c, "benchmark_weights", {}) or {}).items())
        )

        # Tax
        self.tax_enabled.setChecked(bool(c.tax.enabled))
        self.tax_st.setValue(float(c.tax.short_term_rate))
        self.tax_lt.setValue(float(c.tax.long_term_rate))
        self.tax_state.setValue(float(c.tax.state_rate))

        # Planning
        self.plan_enabled.setChecked(bool(c.plan.enabled))
        self.plan_horizon.setValue(int(c.plan.horizon_years))
        self.plan_contrib.setValue(float(c.plan.annual_contribution))
        self.plan_withdraw.setValue(float(c.plan.annual_withdrawal))
        self.plan_goal.setValue(float(c.plan.goal_amount))
        self.plan_inflation.setValue(float(c.plan.inflation))
        self.plan_exp_return.setValue(float(c.plan.expected_return))

        self.error_label.setVisible(False)
        self.configChanged.emit()

    def reset_defaults(self) -> None:
        """New Portfolio: restore defaults and clear the ticker/weights/cost-basis entry."""
        self.load_config(self._defaults)
        self.tickers_edit.clear()
        self.equal_weights.setChecked(True)
        self.weights_group.setVisible(False)
        self.cost_basis_edit.clear()
        self.benchmark_blend.clear()
        self.bl_views_edit.clear()
        self.error_label.setVisible(False)

    def import_holdings_csv(self, path) -> int:
        """Populate tickers (and weights, if a Weight column exists) from a CSV.
        Returns the number of tickers imported; raises on a malformed file."""
        import pandas as pd

        df = pd.read_csv(path)
        cols = {str(col).lower().strip(): col for col in df.columns}
        tcol = next((cols[k] for k in ("ticker", "symbol", "tickers") if k in cols), None)
        if tcol is None:
            raise ValueError("CSV needs a 'Ticker' (or 'Symbol') column.")
        tickers = [str(t).strip().upper() for t in df[tcol].dropna() if str(t).strip()]
        if not tickers:
            raise ValueError("No tickers found in the CSV.")
        self.tickers_edit.setPlainText("\n".join(tickers))

        wcol = next((cols[k] for k in ("weight", "weights", "allocation") if k in cols), None)
        if wcol is None:
            self.equal_weights.setChecked(True)
            self.weights_group.setVisible(False)
            return len(tickers)

        self.equal_weights.setChecked(False)
        self.weights_group.setVisible(True)
        self._rebuild_weights()
        wmap: dict[str, float] = {}
        for t, w in zip(df[tcol], df[wcol]):
            try:
                wmap[str(t).strip().upper()] = float(w)
            except (TypeError, ValueError):
                continue
        total = sum(wmap.values())
        if total > 1.5:  # looks like percentages — normalize to fractions
            wmap = {k: v / total for k, v in wmap.items()}
        for t, spin in self._weight_spins.items():
            if t in wmap:
                spin.setValue(wmap[t])
        self._update_sum()
        return len(tickers)

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
            benchmark_weights=self._parse_benchmark_blend(),
            start_date=self.start_date.date().toPython(),
            end_date=self.end_date.date().toPython(),
            capital=float(self.capital.value()),
            cash=float(self.cash_input.value()) if self._alloc_is_shares() else 0.0,
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
                views=self._parse_bl_views(),
            ),
            backtest=BacktestConfig(
                inception_mode=self.inception_mode.currentData(),
                rebalance_frequency=self.rebalance_freq.currentData(),
                transaction_cost_bps=float(self.cost_bps.value()),
            ),
            cost_basis=self._parse_cost_basis(),
            shares=self._shares_dict(),
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
