# Portfolio Analyzer v2 — Claude Code Implementation Guide

## Context
You are refactoring a Streamlit-based portfolio analysis tool. The old codebase works but has architectural problems: duplicate config keys, subprocess-based execution, matplotlib-only static charts, and no data validation.

The new foundation has been built in `src/`:
- `src/config/models.py` — Pydantic config schema with validation
- `src/data/fetcher.py` — yfinance-only data layer with parquet cache
- `src/data/transforms.py` — Pure return/risk computation functions
- `src/pipeline.py` — Structured pipeline replacing main.py
- `.streamlit/config.toml` — Dark theme

The existing analytics modules (`analytics.py`, `black_litterman.py`, `hierarchical_risk_parity.py`, etc.) contain working math that should be **reused**, not rewritten. The refactor focuses on architecture, IO, and UI — not the core math.

---

## Phase 1: Wire Up the Pipeline (do this first)

### Task 1.1: Make pipeline.py importable alongside old modules
The pipeline imports from `analytics.py`, `performance_attribution.py`, etc. which sit at the project root. Ensure `sys.path` or the project structure allows these imports.

**Option A (quick):** Keep old modules at root, pipeline does `from analytics import max_sharpe`.
**Option B (clean):** Copy analytics modules into `src/analytics/` and update imports.

Recommend Option A for now, refactor later.

### Task 1.2: Build the new app.py
Replace the existing `app.py` with a modern Streamlit app that:

1. **Calls pipeline directly** — no `subprocess.run`
2. **Uses `st.tabs`** for organized output sections
3. **Uses `st.metric` cards** for headline stats
4. **Uses Plotly** for interactive charts
5. **Has a dark theme** via `.streamlit/config.toml` + custom CSS

Layout:
```
┌──────────────────────────────────────────────┐
│  📊 Portfolio Analyzer                [Run]  │
├─────────────┬────────────────────────────────┤
│  SIDEBAR    │  MAIN AREA                     │
│             │                                │
│  Tickers    │  ┌──────────────────────────┐  │
│  Weights    │  │ Metric Cards Row         │  │
│  Dates      │  │ Return | Sharpe | MaxDD  │  │
│  Settings   │  └──────────────────────────┘  │
│  BL Views   │                                │
│             │  [Overview|Risk|Attribution|    │
│             │   Optimization|Forecast]        │
│             │                                │
│  [Run ▶]    │  [Interactive Plotly charts]   │
└─────────────┴────────────────────────────────┘
```

### Task 1.3: Config migration
On startup, if `config.json` exists in old format, use `PortfolioConfig.from_legacy()` to load it. The sidebar writes back in new format via `config.save()`.

---

## Phase 2: Plotly Charts

### Task 2.1: Create `src/charts/plotly_charts.py`
Each function takes typed data (from `AnalysisResults`) and returns a `plotly.graph_objects.Figure`.

```python
def growth_chart(portfolios: dict[str, pd.Series], capital: float) -> go.Figure:
    """Growth of $1M: Active vs Passive vs ORP vs Complete."""

def efficient_frontier_chart(
    frontier_vols: np.ndarray,
    frontier_returns: np.ndarray,
    orp_vol: float,
    orp_return: float,
    rf: float,
) -> go.Figure:
    """Interactive efficient frontier with CAL line and ORP marker."""

def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Interactive correlation heatmap with hover values."""

def drawdown_chart(portfolios: dict[str, pd.Series]) -> go.Figure:
    """Drawdown curves for all portfolios."""

def attribution_chart(df: pd.DataFrame) -> go.Figure:
    """Brinson-Fachler stacked bar chart."""

def capm_scatter(ticker: str, asset_excess: pd.Series, market_excess: pd.Series, alpha: float, beta: float) -> go.Figure:
    """CAPM scatter with regression line."""

def weights_bar(weights: pd.Series, title: str) -> go.Figure:
    """Horizontal bar chart of portfolio weights."""

def forecast_fan(realized: pd.Series, paths: np.ndarray, dates: pd.DatetimeIndex) -> go.Figure:
    """Fan chart for Monte Carlo scenarios."""
```

Design rules:
- Dark theme: `template="plotly_dark"` with custom colors
- Color palette: `["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]`
- Consistent font sizes, axis formatting
- `$` formatting on y-axes for value charts
- `%` formatting for return/risk charts
- No chart borders, subtle gridlines with alpha=0.15

### Task 2.2: Keep matplotlib for PDF reports
The `generate_report.py` module needs static PNGs for PDF. Keep `plotting.py` for that purpose only. Plotly is for the Streamlit app.

---

## Phase 3: Streamlit App Sections

### Overview Tab
- Metric cards: Total Return, Annualized Return, Sharpe, Max Drawdown, Alpha, Beta
- Growth chart (Plotly)
- Active vs Passive cumulative outperformance line

### Risk Tab
- Drawdown chart (Plotly)
- VaR/CVaR histogram
- Rolling volatility & Sharpe (Plotly line charts)
- Correlation heatmap (Plotly)
- Risk metrics table

### Attribution Tab
- Asset-level Brinson-Fachler bar chart
- Sector-level attribution (if available)
- Top/bottom contributors table

### Optimization Tab
- Efficient frontier with ORP marked (Plotly)
- ORP weights bar chart
- HRP weights comparison
- Black-Litterman weights (if enabled)
- Complete portfolio pie/donut

### Forecast Tab
- Monte Carlo fan charts
- Terminal value distribution
- Percentile table (P10/P25/P50/P75/P90)

### Data Tab (collapsible)
- Holdings table
- CAPM regression table
- Factor regression tables
- CSV downloads
- Full report download (ZIP)

---

## Phase 4: Custom CSS for Bloomberg-style Dark Theme

```python
# In app.py, inject after st.set_page_config
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0B1120; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #151D2E 0%, #1A2438 100%);
        border: 1px solid #2D3A50;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Positive/negative delta colors */
    div[data-testid="stMetricDelta"] svg { display: none; }
    div[data-testid="stMetricDelta"] div {
        font-weight: 600 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0B1120;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #151D2E;
        border-radius: 8px;
        padding: 8px 16px;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A5F;
        color: #3B82F6;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0D1526;
        border-right: 1px solid #1E293B;
    }

    /* Tables */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
```

---

## Phase 5: Testing

### `tests/test_config.py`
```python
def test_valid_config_loads():
    cfg = PortfolioConfig(
        tickers=["AAPL", "MSFT"],
        weights={"AAPL": 0.6, "MSFT": 0.4},
        start_date="2020-01-01",
        end_date="2024-12-31",
    )
    assert cfg.allocation_bounds == (0.0, 1.0)

def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="Weights sum to"):
        PortfolioConfig(
            tickers=["AAPL"], weights={"AAPL": 0.5},
            start_date="2020-01-01", end_date="2024-12-31",
        )

def test_legacy_config_migration():
    cfg = PortfolioConfig.from_legacy("config.json")
    assert cfg.benchmark in ("SPY", "^GSPC")
    assert sum(cfg.weights.values()) == pytest.approx(1.0, abs=0.01)
```

### `tests/test_transforms.py`
```python
def test_sharpe_ratio():
    # Known returns -> known Sharpe
    rets = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
    s = T.sharpe_ratio(rets, rf_annual=0.02, periods_per_year=252)
    assert isinstance(s, float)
    assert not np.isnan(s)

def test_max_drawdown():
    values = pd.Series([100, 110, 90, 95, 80, 100])
    assert T.max_drawdown(values) == pytest.approx(-0.2727, abs=0.01)
```

---

## Implementation Order (for Claude Code sessions)

### Session 1: Get it running
1. Copy old analytics modules to project root (keep alongside new src/)
2. Build minimal `app.py` that uses pipeline directly
3. Verify it runs end-to-end with the existing config.json
4. Add the dark theme CSS

### Session 2: Plotly charts
1. Build `src/charts/plotly_charts.py` with all chart functions
2. Wire into app.py tabs
3. Test each chart with real data

### Session 3: UI polish
1. Metric cards with delta indicators
2. Sidebar weight editor with visual preview
3. Progress bar during analysis
4. Download buttons (ZIP, PDF, individual CSVs)

### Session 4: Enhanced analytics
1. HRP integration into optimization tab
2. BL integration with view editor
3. Rolling metrics with Plotly
4. Stress testing module

### Session 5: Reports & testing
1. Split generate_report.py into interpreter + builder
2. Add HTML report option
3. Write test suite
4. Documentation

---

## Files to NOT touch (reuse as-is)
- `analytics.py` — core optimization math works fine
- `black_litterman.py` — BL math is correct
- `hierarchical_risk_parity.py` — HRP algorithm is solid
- `valuation.py` — bond/stock math, standalone
- `factor_loader.py` — Kenneth French data loading works

## Files to REPLACE
- `app.py` — completely new UI
- `main.py` — replaced by `src/pipeline.py`
- `config.json` — new format via Pydantic
- `data_io.py` — replaced by `src/data/fetcher.py`

## Files to REFACTOR later (Phase 3+)
- `generate_report.py` — split into modules
- `make_performance_plots.py` — Plotly equivalents
- `make_additional_plots.py` — Plotly equivalents
- `performance_attribution.py` — clean up, type
- `simulate_forecasts.py` — add bootstrap method
