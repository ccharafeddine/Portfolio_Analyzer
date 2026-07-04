"""Plain-English explanations for every result, so anybody can understand the app.

Each entry has: ``title`` (name), ``what`` (what it is), ``how`` (how to read it),
``why`` (why it matters). ``tooltip_html`` renders a compact rich-text popover for
an info icon; ``inline_text`` renders a one-line beginner blurb.

Beginner mode (a global toggle, offered on first run) controls whether the inline
blurbs are shown throughout the tabs. Info icons are always available.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSettings

from .settings import APP_NAME, ORG_NAME

# ── Content registry ──────────────────────────────────────────────
# Keyed by a stable id used at the call site.
EXPLANATIONS: dict[str, dict[str, str]] = {
    # Headline metrics
    "total_return": {
        "title": "Total Return",
        "what": "The annualized return of your active portfolio — the average yearly growth rate over the period.",
        "how": "Higher is better. The small +/- figure next to it is how much you beat (or lagged) the benchmark.",
        "why": "It's the headline 'how much did I make' number, put on a per-year basis so periods of different lengths are comparable.",
    },
    "sharpe": {
        "title": "Sharpe Ratio",
        "what": "Return earned per unit of risk (volatility), above the risk-free rate.",
        "how": "Higher is better. Rough guide: below 1 is modest, above 1 is good, above 2 is excellent.",
        "why": "Two portfolios can have the same return but very different risk. Sharpe rewards returns that don't come from wild swings.",
    },
    "max_drawdown": {
        "title": "Maximum Drawdown",
        "what": "The largest peak-to-trough drop the portfolio suffered over the period.",
        "how": "Shown as a negative percent — closer to zero is better. -30% means it fell 30% from a high point before recovering.",
        "why": "It's the 'worst pain' number: how deep a hole you'd have sat through. Big drawdowns are hard to hold emotionally.",
    },
    "volatility": {
        "title": "Volatility",
        "what": "How much the portfolio's returns bounce around, annualized (standard deviation).",
        "how": "Lower means a smoother ride. It is not good or bad by itself — it's the 'risk' side of risk-vs-return.",
        "why": "Volatility is the most common measure of risk and feeds into Sharpe and many other statistics.",
    },
    "alpha": {
        "title": "Alpha",
        "what": "Return your portfolio earned beyond what its market exposure (beta) alone would explain, annualized.",
        "how": "Positive alpha means you added value versus just riding the market; negative means you gave some up.",
        "why": "It isolates skill/selection from simply being exposed to the market's ups and downs.",
    },
    "beta": {
        "title": "Beta",
        "what": "How much your portfolio moves when the benchmark moves.",
        "how": "1.0 moves with the market; above 1 is more sensitive (amplified), below 1 is more defensive.",
        "why": "It tells you how much of your risk is just 'the market' versus something specific to your holdings.",
    },
    # Charts
    "growth_chart": {
        "title": "Growth of Capital",
        "what": "How your starting capital would have grown over time in each strategy.",
        "how": "Each line is one portfolio; higher and steeper is more growth. Compare the shapes, not just the endpoints.",
        "why": "It's the most intuitive view of performance — the actual dollar journey, including the bumps along the way.",
    },
    "outperformance_chart": {
        "title": "Cumulative Outperformance",
        "what": "The running dollar gap between your active portfolio and the passive benchmark.",
        "how": "It starts at $0 (no gap at the start). Rising green = you're ahead; falling red = you're behind.",
        "why": "It answers 'is active management actually adding dollars versus just buying the benchmark?'",
    },
    "drawdown_chart": {
        "title": "Drawdown",
        "what": "How far each portfolio is below its own prior peak, at every point in time.",
        "how": "The line sits at 0% at new highs and dips down during declines. Deeper and longer dips are worse.",
        "why": "It shows not just how big losses got, but how long you'd have waited underwater to recover.",
    },
    "return_distribution": {
        "title": "Return Distribution & VaR",
        "what": "A histogram of daily returns, with Value-at-Risk (VaR) and Conditional VaR marked.",
        "how": "Most days cluster near the middle. VaR (95%) is a bad-but-plausible daily loss; CVaR is the average loss on the worst days.",
        "why": "It shows the shape of risk — especially the fat left tail of rare, large losses that averages hide.",
    },
    "correlation_heatmap": {
        "title": "Correlation Heatmap",
        "what": "How closely each pair of holdings moves together.",
        "how": "Near +1 (warm) means they move together; near 0 is independent; negative (cool) means they offset.",
        "why": "Diversification comes from combining assets that don't all move together — low correlations reduce portfolio risk.",
    },
    "stress_test": {
        "title": "Stress Tests",
        "what": "How the portfolio would have done during specific historical crisis periods.",
        "how": "Each bar is one scenario (e.g. a past crash). More negative means the portfolio was hit harder.",
        "why": "Averages hide tail events; stress tests show behavior in the exact moments risk matters most.",
    },
    "scenario": {
        "title": "Scenario Analysis",
        "what": "A forward-looking what-if: your estimated portfolio move if chosen drivers were shocked by set amounts (e.g. equities −20%, oil −30%).",
        "how": "Set a shock (%) for any driver. The estimate is beta × shock summed across drivers — macro factors use your portfolio's historical sensitivity; holdings use their weight. The chart shows each driver's contribution.",
        "why": "Stress tests replay the past; this lets you price *your own* hypotheticals, so you can pressure-test the specific risks you're worried about.",
    },
    "correlation_regime": {
        "title": "Correlation Regime",
        "what": "How the average correlation among your holdings changes over time.",
        "how": "Higher means holdings are moving together more; spikes often happen in market stress.",
        "why": "Diversification can quietly disappear right when you need it — this shows when that's happening.",
    },
    "rolling_metrics": {
        "title": "Rolling Metrics",
        "what": "Key statistics (like return and volatility) computed over a moving window instead of the whole period.",
        "how": "Watch the trend: rising volatility or falling returns show conditions changing over time.",
        "why": "A single number for the whole period hides how performance and risk evolved along the way.",
    },
    "attribution_assets": {
        "title": "Attribution by Asset",
        "what": "How much each holding added to or subtracted from performance versus the benchmark.",
        "how": "Bars above zero helped; below zero hurt. Longer bars had a bigger impact.",
        "why": "It shows where your results actually came from — which picks drove returns and which dragged.",
    },
    "attribution_sectors": {
        "title": "Attribution by Sector",
        "what": "The same performance breakdown, grouped by sector instead of individual holding.",
        "how": "Positive bars are sectors that helped; negative ones hurt, relative to the benchmark.",
        "why": "It reveals whether your edge (or shortfall) came from sector bets versus specific stock picks.",
    },
    "sector_weights": {
        "title": "Sector Allocation",
        "what": "How your capital is spread across market sectors.",
        "how": "Bigger slices are bigger bets. Compare to how concentrated or balanced you want to be.",
        "why": "Sector concentration is a hidden risk — a downturn in one heavy sector can dominate results.",
    },
    "factor_tilts": {
        "title": "Factor Tilts",
        "what": "How much your portfolio leans toward well-known style factors (value, growth, size, momentum, etc.).",
        "how": "Bars show the direction and size of each lean versus a neutral market.",
        "why": "Much of long-run return and risk comes from factor exposures, often without investors realizing it.",
    },
    "attribution_contribution": {
        "title": "Contribution to Active Return by Holding",
        "what": "Each holding's total contribution to your out- or under-performance versus the benchmark over the whole period.",
        "how": "Contribution = the holding's weight × its excess return vs the benchmark. Green added to active return, red dragged; the bars sum to your total active return.",
        "why": "Pinpoints which positions actually earned your edge over the benchmark — or quietly cost you.",
    },
    "attribution_timeseries": {
        "title": "Contribution to Active Return Over Time",
        "what": "How each holding drove your out- or under-performance versus the benchmark as it accumulated over the period.",
        "how": "Each line is a holding's running contribution (its weight times its excess return vs the benchmark); the bold line is total active return. Rising = adding to outperformance, falling = dragging.",
        "why": "A single point-in-time number hides the journey — this shows which holdings earned (or lost) the active return, and when.",
    },
    "capm_scatter": {
        "title": "CAPM Scatter Plots",
        "what": "Each holding's returns plotted against the benchmark's returns, with a fitted line.",
        "how": "The line's slope is beta (market sensitivity); where it crosses is alpha. Tighter clouds mean a stronger fit.",
        "why": "It visually separates how much of a holding's movement is 'the market' versus something specific.",
    },
    "factor_regression": {
        "title": "Factor Regression",
        "what": "A model explaining returns using multiple factors (e.g. Fama-French, momentum, quality).",
        "how": "Each bar is a factor loading — how strongly returns depend on that factor. Bigger means more exposure.",
        "why": "It gives a fuller picture than market-only CAPM of what's really driving your returns.",
    },
    "factor_ff3": {
        "title": "FF3 — Fama-French 3-Factor",
        "what": "The baseline model: market (Mkt-RF), size (SMB, small minus big), and value (HML, high book-to-market minus low).",
        "how": "Each bar is the beta to that factor. Positive SMB = small-cap tilt; positive HML = value tilt, negative HML = growth tilt.",
        "why": "Adds the size and value styles that market-only CAPM misses — the standard starting point for factor analysis.",
    },
    "factor_carhart4": {
        "title": "Carhart 4-Factor",
        "what": "FF3 with one addition: Momentum (MOM) — recent winners minus recent losers.",
        "how": "Read it like FF3, plus the MOM bar: positive means the holding rides momentum (recent winners), negative means it leans contrarian.",
        "why": "Momentum is a strong, persistent return driver that FF3 leaves out; this isolates it on top of size and value.",
    },
    "factor_ff5": {
        "title": "FF5 — Fama-French 5-Factor",
        "what": "FF3 plus two quality dimensions: Profitability (RMW, robust minus weak) and Investment (CMA, conservative minus aggressive).",
        "how": "Read it like FF3, plus RMW (positive = tilt to profitable firms) and CMA (positive = tilt to conservatively-investing firms).",
        "why": "Captures quality and capital-discipline effects that explain returns beyond size and value.",
    },
    "income_metrics": {
        "title": "Income Metrics",
        "what": "Summary of the cash income (dividends) your holdings generate.",
        "how": "Portfolio yield is income as a percent of value; yield-on-cost is income versus what you paid.",
        "why": "For income-focused investors, cash flow matters as much as price gains — this quantifies it.",
    },
    "income_by_position": {
        "title": "Income by Position",
        "what": "How much dividend income each holding contributes.",
        "how": "Taller bars are bigger income sources. Watch for over-reliance on one or two payers.",
        "why": "It shows how concentrated (or diversified) your income stream is across holdings.",
    },
    "cumulative_income": {
        "title": "Cumulative Income",
        "what": "Total dividend income accumulated over time.",
        "how": "A steeper, steadier climb means more reliable, growing income.",
        "why": "It shows the compounding cash a portfolio throws off, separate from price appreciation.",
    },
    "efficient_frontier": {
        "title": "Efficient Frontier",
        "what": "The set of portfolios giving the best possible return for each level of risk.",
        "how": "The curve is the frontier; the marked point is the optimal (max-Sharpe) portfolio. The straight line is the risk/return you can get by mixing it with cash.",
        "why": "It's the core of modern portfolio theory — it shows whether your mix is efficient or you can do better.",
    },
    "orp_weights": {
        "title": "Optimal Portfolio Weights",
        "what": "The holdings mix of the optimal risky portfolio (the max-Sharpe point on the frontier).",
        "how": "Bars are target weights. Compare them to your current weights to see suggested tilts.",
        "why": "It shows the allocation the optimizer thinks gives the best risk-adjusted return.",
    },
    "complete_portfolio": {
        "title": "Complete Portfolio",
        "what": "Your optimal risky portfolio blended with a risk-free asset (like cash or T-bills).",
        "how": "The risk-free slice dials total risk down; more in the risky part means higher expected return and risk.",
        "why": "It shows how to scale risk to your comfort level without changing the underlying holdings mix.",
    },
    "hrp_weights": {
        "title": "Hierarchical Risk Parity",
        "what": "An alternative allocation that groups similar assets and balances risk across the groups.",
        "how": "Weights come from asset relationships rather than return forecasts, so they're often more stable.",
        "why": "It avoids the extreme, fragile bets classic optimization can produce, especially with noisy data.",
    },
    "dendrogram": {
        "title": "Asset Clustering (Dendrogram)",
        "what": "A tree showing which holdings behave most alike.",
        "how": "Assets joined lower down are more similar; higher joins are more distinct groups.",
        "why": "It reveals the real diversification structure — holdings in the same cluster don't diversify each other much.",
    },
    "weight_drift": {
        "title": "Weight Drift",
        "what": "How your holdings' weights wander away from their targets as prices move.",
        "how": "Widening bands mean bigger drift — winners grow and quietly become oversized bets.",
        "why": "Drift changes your risk without any decision on your part; it's the case for periodic rebalancing.",
    },
    "risk_contribution": {
        "title": "Risk Contribution",
        "what": "How much each holding contributes to total portfolio risk (not just its weight).",
        "how": "A holding can be a small weight but a big risk contributor if it's volatile or correlated.",
        "why": "It shows where your risk really sits, which is often very different from where your dollars sit.",
    },
    "concentration": {
        "title": "Concentration Metrics",
        "what": "How spread out or concentrated your portfolio is (HHI, effective number of bets, top-holdings share).",
        "how": "Fewer effective bets or a high top-3 share means more concentration — more upside but more risk.",
        "why": "Concentration is a key risk lever; these numbers make 'how diversified am I really?' concrete.",
    },
    "monte_carlo": {
        "title": "Monte Carlo Forecast",
        "what": "Thousands of simulated future paths for the portfolio, shown as a fan of outcomes.",
        "how": "The middle band is the typical range; the outer edges are optimistic and pessimistic cases.",
        "why": "The future isn't a single line — this shows the range of plausible outcomes and their odds.",
    },
    "probability": {
        "title": "Probability Analysis",
        "what": "Odds and outcome levels from the simulations (expected value, chance of loss, best/worst cases).",
        "how": "P(Loss) is the chance of ending below where you started; P5/P50/P95 are pessimistic/median/optimistic ends.",
        "why": "It turns the simulation into concrete, decision-useful probabilities instead of just a picture.",
    },
    "coverage": {
        "title": "Data Coverage",
        "what": "When each holding's price history begins within your window.",
        "how": "Each bar starts at the asset's first trading day. A short bar means the asset is newer than the window (e.g. a recent IPO).",
        "why": "It makes the backtest honest: you can see exactly which holdings weren't present for the whole period and were phased in.",
    },
    "dual_window": {
        "title": "Full History vs Common Window",
        "what": "The same metrics computed two ways when your holdings have different start dates.",
        "how": "'Full History' uses everything available (composition changes over time); 'Common Window' only covers the period where every holding exists (your true target portfolio).",
        "why": "Full history maximizes data, but only the common window reflects the exact portfolio you configured — comparing both keeps you honest.",
    },
    "capture": {
        "title": "Up / Down Capture",
        "what": "How much of the benchmark's gains (up) and losses (down) your portfolio captured.",
        "how": "Up capture above 100% means you beat the benchmark in rising markets; down capture below 100% means you fell less in declines. High up + low down is ideal.",
        "why": "It separates 'good' risk (participating in gains) from 'bad' risk (participating in losses) — two portfolios with the same return can differ hugely here.",
    },
    "rolling_alpha_beta": {
        "title": "Rolling Alpha & Beta",
        "what": "Your alpha (skill) and beta (market sensitivity) measured over a moving window instead of once.",
        "how": "Watch the trend: rising beta means you're taking more market risk over time; positive, stable alpha suggests consistent value-add.",
        "why": "A single alpha/beta hides how your exposures drifted — this shows whether performance came from skill or from dialing risk up and down.",
    },
    "trading_costs": {
        "title": "Trading & Costs",
        "what": "How often the portfolio rebalanced, how much it traded, and the total transaction cost.",
        "how": "More rebalances and higher turnover mean more trading cost — which drags on returns. Compare against the benefit of staying near target weights.",
        "why": "Costs are a real, controllable drag; seeing them makes rebalancing a deliberate trade-off rather than a hidden expense.",
    },
    "tax": {
        "title": "Tax Analysis",
        "what": "Unrealized gains/losses on your current holdings, positions you could sell at a loss to offset taxes (harvesting), and the estimated tax on gains realized by rebalancing.",
        "how": "Harvestable losses are dollars of unrealized loss you could realize to reduce your tax bill. Estimated tax applies your short/long-term rates to gains that rebalancing sold.",
        "why": "Taxes are one of the biggest drags on real, after-tax returns — and one of the few you can actively manage.",
    },
    "retirement_plan": {
        "title": "Retirement Plan",
        "what": "Thousands of simulated futures for your portfolio with your contributions and withdrawals applied over time.",
        "how": "The fan shows the range of outcomes; the middle band is typical. Paths that hit zero mean the money ran out before the horizon.",
        "why": "It answers the real question — 'will my money last?' — as a probability, not a single guess.",
    },
    "plan_metrics": {
        "title": "Plan Metrics",
        "what": "The odds and outcomes of your plan.",
        "how": "Success probability is your chance of not running out (or reaching your goal). Safe withdrawal rate is the most you could take out yearly and still succeed ~90% of the time.",
        "why": "These turn the projection into concrete, decision-useful numbers you can act on.",
    },
    "performance_table": {
        "title": "Performance Summary",
        "what": "Headline return and risk statistics for each portfolio side by side.",
        "how": "Compare annualized return, volatility, Sharpe, and max drawdown across the active portfolio and the benchmark.",
        "why": "It's the at-a-glance scorecard — how much you made and how much risk you took to make it.",
    },
    "tail_risk_metrics": {
        "title": "Tail Risk Metrics",
        "what": "Drawdown and downside statistics that describe the bad-case behavior of the portfolio.",
        "how": "Larger drawdowns and worse VaR/CVaR mean deeper potential losses. Compare across portfolios.",
        "why": "Averages hide the pain; these numbers describe how bad things can get, which is what actually tests your resolve.",
    },
    "extended_risk": {
        "title": "Extended Risk Statistics",
        "what": "A fuller set of risk ratios — Sortino, Calmar, skewness, kurtosis, best/worst day, and more.",
        "how": "Sortino and Calmar reward return per unit of downside risk; negative skew and high kurtosis flag fat-tailed, crash-prone return patterns.",
        "why": "They round out the risk picture beyond volatility, capturing asymmetry and tail behavior.",
    },
    "capm": {
        "title": "CAPM Regression",
        "what": "Each holding's alpha (skill) and beta (market sensitivity) versus the benchmark, with statistical significance.",
        "how": "Beta near 1 moves with the market; positive alpha with a high t-stat suggests genuine, reliable outperformance.",
        "why": "It separates returns that came from market exposure from returns that came from selection.",
    },
    "rebalancing": {
        "title": "Rebalanced vs Buy-and-Hold",
        "what": "How periodic rebalancing compares to simply letting your holdings drift.",
        "how": "Compare return, volatility, and drawdown. Rebalancing usually lowers risk; it may raise or lower return.",
        "why": "It shows whether disciplined rebalancing actually helped, net of the trading it required.",
    },
    "turnover": {
        "title": "Turnover",
        "what": "How much of the portfolio was traded at each rebalance.",
        "how": "Higher turnover means more trading — and more cost and potential taxes. Watch for spikes.",
        "why": "Turnover is the hidden price of rebalancing; keeping an eye on it keeps costs in check.",
    },
    "trade_recommendations_target": {
        "title": "Portfolio Rebalancing Trades",
        "what": "The exact buy/sell orders to move your holdings from where they've drifted back to the target weights you specified.",
        "how": "Each row shows current vs your target weight, the dollar trade (+buy / −sell), and an approximate share count. Green bars buy, red bars sell.",
        "why": "Rebalancing back to your intended allocation controls risk drift and enforces buy-low/sell-high discipline — a concrete to-do list instead of a spreadsheet exercise.",
    },
    "trade_recommendations_orp": {
        "title": "ORP Rebalancing Trades",
        "what": "The buy/sell orders to move your holdings to the Optimal Risky Portfolio (ORP) — the max-Sharpe weights the optimizer computed.",
        "how": "Same layout, but the target is the ORP's optimized weights rather than your own. Trades can be large when the ORP differs a lot from your current mix.",
        "why": "Shows what it would actually take to adopt the mean-variance-optimal allocation, so you can weigh the move — and its turnover — deliberately.",
    },
    "orp_stats": {
        "title": "Optimal Risky Portfolio",
        "what": "The expected return, volatility, and Sharpe of the max-Sharpe optimal portfolio.",
        "how": "This is the theoretical best risk-adjusted mix from mean-variance optimization, for reference.",
        "why": "It's the benchmark your actual allocation is measured against on the efficient frontier.",
    },
    "simulation_comparison": {
        "title": "Simulation Comparison",
        "what": "Side-by-side outcomes from the different Monte Carlo methods (parametric vs bootstrap).",
        "how": "Compare expected value and percentile outcomes. Agreement across methods increases confidence.",
        "why": "Each method makes different assumptions; seeing them together shows how sensitive the forecast is to method.",
    },
    "historical_risk": {
        "title": "Historical Risk Statistics",
        "what": "Backward-looking risk stats (gain-to-pain, daily VaR/CVaR) for the active portfolio.",
        "how": "More negative VaR/CVaR means larger typical bad-day losses; higher gain-to-pain is better.",
        "why": "A grounding in what actually happened, alongside the forward-looking simulations.",
    },
    "reports": {
        "title": "Reports",
        "what": "Downloadable, shareable summaries of the full analysis — HTML, PDF, and a client-ready PowerPoint.",
        "how": "Click a format to generate and save it. The PowerPoint is a polished, presentation-ready deck.",
        "why": "Reports let you hand off the analysis to a client or colleague without them needing the app.",
    },
    "holdings": {
        "title": "Holdings",
        "what": "The opening positions the backtest bought — ticker, shares, price, dollars invested, and realized weight.",
        "how": "Realized weight is the actual share of capital each holding took once whole/fractional shares were bought.",
        "why": "It's the concrete starting portfolio behind every number in the app.",
    },
    "run_config": {
        "title": "Run Configuration",
        "what": "The exact settings this analysis used, as saved JSON.",
        "how": "Every input — universe, weights, dates, and advanced options — is captured for reproducibility.",
        "why": "It lets you (or someone else) reproduce the exact run later, or audit what was assumed.",
    },
    "fundamentals": {
        "title": "Fundamentals",
        "what": "Company-level financial metrics for each holding — valuation (P/E, P/B), profitability (margins, ROE), growth, balance-sheet health, and dividends.",
        "how": "Compare holdings side by side. Lower P/E and higher margins/ROE are generally more attractive, but interpret within each sector. A DCF fair value above the price suggests potential undervaluation.",
        "why": "Price performance tells you what happened; fundamentals tell you what you own and whether the valuation is justified.",
    },
    "financial_statements": {
        "title": "Financial Statements",
        "what": "Annual income statement, balance sheet, and cash-flow history for the selected holding, plus analyst price targets and the buy/hold/sell mix.",
        "how": "Look for consistent revenue and free-cash-flow growth, a manageable debt load, and where the price sits versus the analyst mean target.",
        "why": "The comparison tables show the current snapshot; the statements show the trajectory and what Wall Street expects next.",
    },
    "earnings_calendar": {
        "title": "Upcoming Earnings & Dividends",
        "what": "The next scheduled earnings reports and ex-dividend dates for your holdings, soonest first.",
        "how": "Earnings dates flag when a stock is likely to move on results. To receive a dividend you must own the shares before the ex-dividend date.",
        "why": "These are the known catalysts on the horizon — useful for anticipating volatility and planning around income.",
    },
    "news_feed": {
        "title": "Latest News",
        "what": "Recent headlines for the holdings in your analysis, newest first.",
        "how": "Click a headline to open the full article. When an Alpha Vantage key is set, each item also shows a sentiment tag (bullish/neutral/bearish).",
        "why": "Prices move on news. Seeing the current narrative around your holdings adds fundamental context the numbers alone don't capture.",
    },
    "rates_macro": {
        "title": "Rates & Treasuries",
        "what": "The current U.S. Treasury yield curve and headline macro rates (Fed Funds, CPI, unemployment), from FRED.",
        "how": "An upward-sloping curve is normal; an inverted one (short yields above long) has historically preceded recessions. Compare the latest level to its one-year change.",
        "why": "Rates are the discount rate for every asset. The macro backdrop shapes valuations, the risk-free rate, and what your portfolio is competing against.",
    },
    "watchlist": {
        "title": "Watchlist",
        "what": "A personal list of symbols you want to track — stocks, ETFs, or crypto — with live (delayed) quotes, kept separate from the portfolio you analyze.",
        "how": "Type a symbol and press Add; crypto shorthand like BTC becomes BTC-USD automatically. Each row shows the latest price and today's change, green when up and red when down. Click the ✕ (or right-click) to remove a row, and the column headers to sort. It refreshes on its own about once a minute while this screen is open, or immediately with Refresh.",
        "why": "It lets you keep an eye on tickers you're considering or already follow without adding them to a portfolio or running a full analysis. The quotes are fetched independently, so nothing here touches your saved portfolios or results.",
    },
    "day_change_heatmap": {
        "title": "Day-Change Heatmap",
        "what": "A tiled grid of your symbols shaded by today's percentage move — green for gains, red for losses, brighter with bigger moves.",
        "how": "Scan for color, not numbers: a wall of green is a broad up day, mostly red a down day. One bright tile against a calm grid flags an outlier mover worth a look.",
        "why": "It turns a table of quotes into an at-a-glance risk picture, so you see where today's action is concentrated without reading every row.",
    },
    "data_exports": {
        "title": "Data Exports",
        "what": "The raw output files (CSVs) from the analysis, individually or as a ZIP.",
        "how": "Export them to work with the numbers in Excel, Python, or another tool.",
        "why": "Full transparency — you own the underlying data, not just the on-screen summary.",
    },
}


# ── Configuration field help (short, one-liner tooltips) ──
CONFIG_HELP: dict[str, str] = {
    "tickers": "The holdings in your portfolio — stock, ETF, or crypto symbols, one per "
    "line (e.g. AAPL, SPY, BTC-USD).",
    "weights": "How your capital is split across holdings. 'Equal weights' splits evenly; "
    "uncheck it to set each by hand (they must add up to 1).",
    "benchmark": "The index or ETF to measure your portfolio against (e.g. SPY for the "
    "S&P 500). Alpha, beta, and outperformance are all relative to this.",
    "bl_views": "Your forward-looking opinions, blended into the optimizer's expected "
    "returns (Black-Litterman). Absolute: 'AAPL: 12%' (AAPL will return ~12%/yr). "
    "Relative: 'AAPL > MSFT: 3%' (AAPL beats MSFT by 3%). Add '@high' or '@low' to set "
    "how strongly the view pulls the result. Only affects the ORP / optimized portfolio.",
    "benchmark_blend": "Optional multi-asset benchmark. Enter 'TICKER: weight' per line "
    "(e.g. 'SPY: 0.6' and 'AGG: 0.4' for a classic 60/40). When set, this fixed-weight "
    "mix replaces the single benchmark, and the Benchmark field above is just its label.",
    "dates": "The historical window to analyze. A longer window captures more market "
    "conditions (bull and bear).",
    "capital": "The starting dollar amount to invest. All growth and income figures scale "
    "from this number.",
    "risk_free": "The annual return of a 'safe' asset like Treasury bills, as a decimal "
    "(0.04 = 4%). Feeds the Sharpe ratio and the optimal-portfolio math.",
    "max_weight": "The largest fraction any single holding may take in the optimized "
    "portfolio. Lower values force more diversification.",
    "orp_fraction": "How much of your money goes into the optimal risky portfolio versus a "
    "risk-free asset. 0.8 means 80% invested, 20% held safe.",
    "allow_shorts": "Allow negative weights (betting a holding will fall). Leave off to keep "
    "every weight positive (long-only), which is normal for most investors.",
    "include_orp": "Compute the optimal risky portfolio (the max-Sharpe mix) and the "
    "efficient frontier. Turn off to skip that optimization.",
    "dividends": "Use dividend-adjusted returns and compute income analytics (yields and "
    "dividend growth) in the Income tab.",
    "bl_tau": "Black-Litterman 'tau': how much to trust your own views versus the market's "
    "implied view. Smaller means lean on the market more.",
    "inception_mode": "How to handle an asset that didn't exist for the whole window (e.g. a "
    "recent IPO). 'Rescale' holds the available assets at rescaled weights and adds the new "
    "one when it starts trading. 'Cash' parks that asset's weight in cash until it exists.",
    "rebalance_frequency": "How often to rebalance back to your target weights. 'Buy & hold' "
    "never rebalances (weights drift with prices); a schedule trades periodically to restore "
    "targets. New assets are always added when they begin trading, regardless of this.",
    "transaction_cost_bps": "A trading cost applied to every rebalance, in basis points of the "
    "dollars traded (1 bp = 0.01%). Models real-world friction; higher costs penalize frequent "
    "rebalancing.",
    "tax_enabled": "Turn on tax-aware analysis: unrealized gains/losses, tax-loss-harvesting "
    "candidates, and estimated tax on gains realized by rebalancing.",
    "tax_short": "The tax rate on short-term gains (assets held under a year) — usually your "
    "ordinary income rate. Used to estimate tax on rebalancing sells.",
    "tax_long": "The tax rate on long-term gains (assets held over a year) — typically lower "
    "than the short-term rate (often 15-20%).",
    "tax_state": "Your state capital-gains tax rate, if any. Added on top of the federal rates.",
    "cost_basis": "Your average purchase price per share for each holding, one per line "
    "(e.g. 'AAPL: 150'). Leave blank to infer it from the backtest's opening price. Drives "
    "unrealized gain/loss and tax estimates.",
    "plan_enabled": "Turn on retirement/withdrawal planning: project the portfolio forward "
    "with contributions and withdrawals to estimate your odds of success.",
    "plan_horizon": "How many years to project forward (e.g. 30 years of retirement).",
    "plan_contribution": "How much you add to the portfolio each year (inflation-adjusted). "
    "Use during the saving/accumulation phase.",
    "plan_withdrawal": "How much you take out each year (inflation-adjusted). Use during the "
    "spending/retirement phase. This is what drives depletion risk.",
    "plan_goal": "A target ending balance. If set, 'success' means finishing at or above this "
    "amount. Leave at 0 to define success simply as not running out of money.",
    "plan_inflation": "Assumed annual inflation. Contributions and withdrawals grow by this "
    "each year so they keep their real (spending-power) value.",
    "plan_expected_return": "The long-run average annual return to assume for the projection. "
    "It keeps your portfolio's historical ups-and-downs (volatility) but recenters the average, "
    "so a few hot recent years aren't extrapolated for decades. 7% is a common long-run assumption.",
}


def config_tooltip(key: str) -> str:
    text = CONFIG_HELP.get(key)
    return f"<div style='max-width:300px'>{text}</div>" if text else ""


def get(key: str) -> Optional[dict]:
    return EXPLANATIONS.get(key)


def tooltip_html(key: str) -> str:
    e = EXPLANATIONS.get(key)
    if not e:
        return ""
    return (
        f"<div style='max-width:320px'>"
        f"<b>{e['title']}</b><br><br>"
        f"{e['what']}<br><br>"
        f"<b>How to read it:</b> {e['how']}<br><br>"
        f"<b>Why it matters:</b> {e['why']}"
        f"</div>"
    )


def inline_text(key: str) -> str:
    e = EXPLANATIONS.get(key)
    if not e:
        return ""
    return f"{e['what']}  {e['how']}"


# ── Beginner mode (global preference) ──────────────────────────────
def is_beginner_mode() -> bool:
    return QSettings(ORG_NAME, APP_NAME).value("beginner_mode", False, type=bool)


def set_beginner_mode(enabled: bool) -> None:
    QSettings(ORG_NAME, APP_NAME).setValue("beginner_mode", bool(enabled))


def is_onboarded() -> bool:
    return QSettings(ORG_NAME, APP_NAME).value("onboarded", False, type=bool)


def set_onboarded() -> None:
    QSettings(ORG_NAME, APP_NAME).setValue("onboarded", True)
