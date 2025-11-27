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
- Attempted PDF report (`outputs/report.pdf`, optional)

This tool is designed so any user can plug in their own tickers, weights, and dates using `config.json` and reproduce the analysis easily.

---

## 📁 Project Structure

PORTFOLIO_ANALYZER/
├── pycache/
├── .venv/ # virtual environment (ignored by git)
├── outputs/ # all generated outputs go here
│ ├── active_portfolio_value.csv
│ ├── passive_portfolio_value.csv
│ ├── complete_portfolio_value.csv
│ ├── orp_value_realized.csv
│ ├── clean_prices.csv
│ ├── holdings_table.csv
│ ├── correlation_matrix.png
│ ├── complete_portfolio_pie.png
│ ├── performance_vs_benchmark.png
│ ├── forward_scenarios.png
│ └── report.md
├── analytics.py
├── build_active_portfolio_series.py
├── build_passive_portfolio_series.py
├── compute_active_stats.py
├── config.json # MAIN CONFIG — EDIT THIS TO RUN NEW ANALYSIS
├── data_io.py
├── generate_report.py
├── main.py # ENTRY POINT — run the analysis
├── make_additional_plots.py
├── make_performance_plots.py
├── plotting.py
├── README.md
├── requirements.txt
├── simulate_forecasts.py
├── style_analysis.py
└── valuation.py



You typically only modify:  
**`config.json`** (portfolio, dates, risk-free rate, etc.)

---

## 🛠️ Requirements

- **Python 3.10+** (project tested on Python 3.12)
- Internet connection (for Yahoo Finance)
- Git (optional)


Install Python packages:

pip install -r requirements.txt


---

## 🚀 How to Run the Project

### **1. Clone and open**

git clone [<your-repo-url>](https://github.com/ccharafeddine/Portfolio_Analyzer).git
cd Portfolio_Analyzer



### **2. Create a virtual environment**

macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate


Windows (PowerShell):

python -m venv .venv
..venv\Scripts\Activate.ps1



### **3. Install dependencies**

pip install -r requirements.txt


### **4. Run the analysis**

python main.py --config config.json


All outputs appear in the `outputs/` directory.

---

## 🧩 Editing `config.json` (The ONLY file needed for new analyses)

Below is a descriptive example of `config.json` and what each field controls.

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
What you modify:
Change tickers → "tickers": [...]

Change benchmark → "benchmark": "^GSPC"

Change analysis window → "start" and "end"

Change risk-free rate → "risk_free_rate": 0.04

Change active portfolio weights → inside "weights"

Update capital / portfolio size

Update style tickers (inside style_analysis.py if desired)

That’s it.
Everything else updates automatically.

What Each Script Does:

main.py — The Workflow Controller
Coordinates everything:

Loads config

Pulls price data

Builds active & passive portfolios

Computes stats, Sharpe, correlations, ORP

Runs regressions & simulations

Generates plots

Builds final report

Run this file to run the full analysis.


data_io.py
Handles:

Downloading price data (yfinance)

Saving/loading:

clean_prices.csv

holdings tables

portfolio value series


build_active_portfolio_series.py
Calculates share counts based on weights

Builds active_portfolio_value.csv

Builds holdings_table.csv


build_passive_portfolio_series.py
Buys benchmark at start_date

Builds passive_portfolio_value.csv


compute_active_stats.py
Calculates:

Annual return & volatility

Sharpe ratio

Correlation matrix

Efficient frontier

Optimal Risky Portfolio (ORP)

Creates:

complete_portfolio_value.csv

orp_value_realized.csv

correlation_matrix.png

complete_portfolio_pie.png


analytics.py
CAPM regression

Excess returns

Math/stat helper functions


make_performance_plots.py
Creates performance plots comparing:

active

passive

ORP

complete portfolio


make_additional_plots.py
Efficient frontier plots

Additional visuals not covered in the performance file


style_analysis.py
Multi-factor “style” regression

You may update the tickers used here to run a different factor analysis


simulate_forecasts.py
Monte Carlo–style forward simulations using historical vol & drift

Produces charts like forward_scenarios.png


plotting.py
Shared utilities for consistent plot formatting


generate_report.py
Reads all CSVs & plots

Builds a complete report saved as:

outputs/report.md

outputs/report.pdf (optional; may fail if FPDF hits formatting limits)


Outputs Produced:
Inside the outputs/ folder you will find:

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

These files collectively describe the entire portfolio analysis.


🔁 Running with a New Portfolio (Quick Guide)
Open config.json

Update:

tickers

weights

benchmark

start/end dates

risk-free rate

capital

complete_portfolio.y


Save


Run:

python main.py --config config.json


You now have a fully regenerated analysis in outputs/.



Notes & Limitations:

Yahoo Finance occasionally fails silently; re-run if price data is missing.

PDF generation may fail due to Unicode/font limitations. Markdown report always works.

Optimization relies on historical returns; results should not be interpreted as forecasts.


Intended Use
This tool was built to allow students, analysts, and researchers to:

Understand portfolio construction

Compare active management vs benchmarks

Explore optimization & risk decomposition

Produce clean reports for academic or professional purposes