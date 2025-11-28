# Portfolio Analyzer

A Python-based toolkit for analyzing an **active multi-asset portfolio** versus a **passive benchmark**.  
This project downloads historical price data, constructs portfolios, computes risk/return statistics, runs regressions and simulations, and generates a complete performance report.

Outputs include:

- Active, passive, ORP, and complete portfolio value series  
- Risk/return statistics  
- CAPM regression results  
- Correlation matrix  
- Efficient frontier / ORP summary  
- Forward return scenarios  
- A full Markdown report (`outputs/report.md`)  
- Optional PDF report (`outputs/report.pdf`)

Any user can plug in their own tickers, weights, and dates using `config.json` to reproduce the analysis.

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
  "tickers": [
    "BTC-USD", "MARA", "SPYG", "XLK", "AAPL",
    "UNH", "XLP", "SCHD", "XYLD", "SHY", "IAU"
  ],

  "benchmark": "^GSPC",

  "start": "2020-01-01",
  "end": null,
  "interval": "1d",

  "use_dividends_in_total_return": false,
  "risk_free_rate": 0.04,

  "max_allocation_bounds": [0.0, 1.0],
  "short_sales": false,
  "frontier_points": 50,

  "active_portfolio": {
    "capital": 1000000,
    "start_date": "2020-01-02",
    "weights": {
      "BTC-USD": 0.08,
      "MARA":    0.02,
      "SPYG":    0.20,
      "XLK":     0.15,
      "AAPL":    0.10,
      "UNH":     0.05,
      "XLP":     0.10,
      "SCHD":    0.10,
      "XYLD":    0.10,
      "SHY":     0.07,
      "IAU":     0.03
    }
  },

  "passive_portfolio": {
    "capital": 1000000,
    "start_date": "2020-01-02"
  },

  "complete_portfolio": {
    "y": 0.80
  }
}
```

What you modify:
```
Change tickers → "tickers": [...]

Benchmark ticker → "benchmark": "^GSPC"

Start/end dates → "start" / "end"

Risk-free rate → "risk_free_rate": 0.04

Active weights → inside "weights"

Portfolio size → "capital"

Complete portfolio mix → "y": 0.80
```

Everything else updates automatically.


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

Inside outputs/:
```
active_portfolio_value.csv

passive_portfolio_value.csv

complete_portfolio_value.csv

orp_value_realized.csv

holdings_table.csv

clean_prices.csv

correlation_matrix.png

complete_portfolio_pie.png

performance_vs_benchmark.png

forward_scenarios.png

report.md

report.pdf (optional)
```

**<u>Running With a New Portfolio (Quick Guide)</u>**

**Open config.json**

**Update:**

tickers

weights

start/end dates

benchmark

risk-free rate

capital

complete portfolio fraction

**Save**

**Run:**

```python
python main.py --config config.json
```
All outputs regenerate automatically.

**<u>Notes & Limitations</u>**
```
Yahoo Finance occasionally drops data — rerun if something seems missing.

PDF export may fail due to Unicode or layout constraints (markdown always succeeds).

Optimization is based on historical returns; not predictive.
```

**<u>Intended Use</u>**
```
This tool helps:

Students learning portfolio theory

Analysts comparing active vs passive strategies

Researchers exploring optimization

Anyone building structured finance reports
```