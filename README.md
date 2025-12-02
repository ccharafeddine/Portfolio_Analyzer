# Portfolio Analyzer

A complete Python + Streamlit toolkit for portfolio optimization, risk analysis, simulations, and automated report generation.

**Overview**

The Portfolio Analyzer is a full-featured system for analyzing:
```
Active portfolios (your chosen tickers + weights)

Passive benchmark portfolios (e.g., SPY, ^GSPC, etc.)

Optimal Risky Portfolio (ORP) via Max-Sharpe optimization

Complete portfolio blending ORP with risk-free assets

Portfolio factor exposures (CAPM + style regressions)

Forward-looking Monte Carlo forecasts

Efficient frontier visualization

Automated full report generation (Markdown + optional PDF)
```

You can now run all analyses through a Streamlit app that:
```
Lets you input tickers, weights, bounds, and settings

Runs the full pipeline with one button

Shows all outputs directly in the browser

Offers instant ZIP download of all plots + CSVs

Automatically sorts outputs into a clean, professional order
```
**Web App**

The app.py provides a complete Streamlit user interface.

Can run from ccportfolioanalyzer.streamlit.app

Features:
```
Text box for tickers (one per line)

Per-ticker weight inputs or equal-weight toggle

Dividend toggle

Shorting toggle

If ON: per-asset bounds = [-1.5, +1.5]

If OFF: per-asset bounds = [0, +1.5]

Start/end date selectors

Benchmark selector

Complete portfolio risk-free mix slider

“Run Analysis” button triggers main.py

Automatic clearing of old outputs between runs

Ordered output display:

ZIP download

Reports

Holdings + CAPM tables

Key charts (efficient frontier, CAL, ORP pie, etc.)

Forward simulation

CSVs

CAPM scatter plots
```
**Deployment**

You can deploy the entire app on Streamlit Cloud, using:
```
FMP API key via st.secrets["FMP_API_KEY"]

Alpha Vantage key via st.secrets["ALPHA_VANTAGE_API_KEY"]
```
Both are supported in data_io.py.

**API Keys (FMP + Alpha Vantage)**

Place your keys in your environment:

Local (Windows PowerShell)
```powershell
setx FMP_API_KEY "YOUR_KEY"
setx ALPA_VANTAGE_API_KEY "YOUR_KEY"
```
Streamlit Cloud (Add to Secrets)
```powershell
FMP_API_KEY = "YOUR_KEY"
ALPHA_VANTAGE_API_KEY = "YOUR_KEY"
```

**Running the Analysis**
Option 1 - Use the Streamlit app
```bash
streamlit run app.py
```
Option 2 - Direct Python
```bash
python main.py --config config.json
```

---

## Project Structure

```text
PORTFOLIO_ANALYZER/
├── pycache/
├── .venv/                   # virtual environment (ignored by git)
├── outputs/                 # all generated outputs go here
│   ├── active_portfolio_value.csv
│   ├── passive_portfolio_value.csv
│   ├── complete_portfolio_value.csv
│   ├── orp_value_realized.csv
│   ├── clean_prices.csv
│   ├── holdings_table.csv
│   ├── correlation_matrix.png
│   ├── complete_portfolio_pie.png
│   ├── performance_vs_benchmark.png
│   ├── forward_scenarios.png
│   └── report.md
├── analytics.py
├── build_active_portfolio_series.py
├── build_passive_portfolio_series.py
├── compute_active_stats.py
├── config.json              # MAIN CONFIG — EDIT THIS TO RUN NEW ANALYSIS
├── data_io.py
├── generate_report.py
├── main.py                  # ENTRY POINT — run the analysis
├── make_additional_plots.py
├── make_performance_plots.py
├── plotting.py
├── README.md
├── requirements.txt
├── simulate_forecasts.py
├── style_analysis.py
└── valuation.py
```
You usually only modify:

`config.json` (portfolio, weights, dates, risk-free rate, etc.)


Requirements:
```
Python 3.10+ (tested on Python 3.12)

Internet connection (for Yahoo Finance)

Git (optional)
```

Install Python packages:
```python
pip install -r requirements.txt
```
How to Run the Project

1. Clone and open
```python
git clone https://github.com/ccharafeddine/Portfolio_Analyzer.git

cd Portfolio_Analyzer
```

2. Create a virtual environment

macOS / Linux:
```python
python3 -m venv .venv

source .venv/bin/activate
```
Windows (PowerShell):
```python
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

3. Install dependencies
```python
pip install -r requirements.txt
```
4. Run the analysis
```python
python main.py --config config.json
```
All outputs appear in the outputs/ directory.

**<u>Editing config.json (The ONLY file needed for new analyses)</u>**

Below is a clear, fully valid example.

```json
{
  "tickers": ["AAPL", "MSFT", "GLD"],
  "benchmark": "SPY",
  "start": "2020-01-01",
  "end": "2025-12-31",

  "short_sales": false,
  "max_allocation_bounds": [0, 1.5],
  "risk_free_rate": 0.04,
  "frontier_points": 50,

  "active_portfolio": {
    "capital": 1000000,
    "tickers": ["AAPL", "MSFT", "GLD"],
    "weights": {"AAPL":0.34,"MSFT":0.33,"GLD":0.33},
    "start_date": "2020-01-01",
    "end_date": "2025-12-31"
  },

  "passive_portfolio": {
    "capital": 1000000,
    "start_date": "2020-01-01"
  },

  "complete_portfolio": {"y": 0.80}
}
```
The UI writes this, no need to edit it manually


**<u>What Each Script Does</u>**

**<u>main.py — Workflow Controller</u>**
```
Coordinates everything:

Loads config

Downloads data

Builds portfolios

Computes statistics

Runs CAPM regression

Simulations

Generates plots

Writes the final report

Run this file to perform the full analysis.
```
**<u>data_io.py</u>**
```
Handles downloading and saving:

Price data (Yahoo Finance)

clean_prices.csv

holdings_table.csv

portfolio value series
```
**<u>build_active_portfolio_series.py</u>**
```
Computes share counts from weights.
Generates:

active_portfolio_value.csv

holdings_table.csv
```
**<u>build_passive_portfolio_series.py</u>**
```
Buys benchmark at start.
Generates:

passive_portfolio_value.csv
```
**<u>compute_active_stats.py</u>**
```
Computes:

Annual returns

Volatility

Sharpe

Correlation matrix

Efficient frontier

Optimal Risky Portfolio (ORP)

Creates:

complete_portfolio_value.csv

orp_value_realized.csv

correlation_matrix.png

complete_portfolio_pie.png
```
**<u>analytics.py</u>**
```
CAPM regression

Excess returns

Math/stat helpers
```
**<u>make_performance_plots.py</u>**
```
Creates charts comparing:

Active

Passive

ORP

Complete
```
**<u>make_additional_plots.py</u>**
```
Efficient frontier and additional visuals.
```
**<u>style_analysis.py</u>**
```
Multi-factor regression (can customize tickers)
```
**<u>simulate_forecasts.py</u>**
```
Monte Carlo-style forward simulations.

Creates:

forward_scenarios.png
```
**<u>plotting.py</u>**
```
Shared utilities for consistent plotting.
```
**<u>generate_report.py</u>**
```
Builds:

report.md (always works)

report.pdf (may fail if FPDF hits Unicode or space limits)
```
**<u>Outputs Generated</u>**

Outputs (Ordered Automaticaly in Web App):
```
Download all outputs (ZIP)

Report files (md, pdf)

Holdings table

CAPM summary table

Efficient frontier

ORP weights table (new!)

CAL plot

ORP vs expected chart

Complete portfolio pie

Growth chart

Forward scenario simulations

Additional charts

CSV file downloads

CAPM scatter plots (end)
```

**<u>Example Portfolio To Try</u>**

```
AAPL
MSFT
NVDA
GLD
TLT
XLF
XLK
BTC-USD
```


**<u>Notes & Limitations</u>**
```
FMP free tier limits historical data depth

Alpha Vantage handles BTC safely

PDF generation may fail if Unicode-heavy

Optimization uses historical returns—not predictive
```

**<u>Intended Use</u>**
```
Finance students (Sharpe, CAPM, frontier)

Wealth management trainees

Analysts comparing active vs passive

Quants testing workflow automation

Anyone wanting fast multi-asset portfolio analytics
```