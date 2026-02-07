"""
Interactive Plotly charts for Portfolio Analyzer v2.

Every function takes typed data and returns a plotly Figure.
Dark theme, consistent styling, hover-rich.

Color palette:
  Primary:   #3B82F6  (blue)
  Success:   #10B981  (green)
  Warning:   #F59E0B  (amber)
  Danger:    #EF4444  (red)
  Purple:    #8B5CF6
  Pink:      #EC4899
  Cyan:      #06B6D4
  Slate:     #94A3B8  (muted text)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ──────────────────────────────────────────────────────────────
# Theme constants
# ──────────────────────────────────────────────────────────────

COLORS = [
    "#3B82F6",  # blue
    "#10B981",  # green
    "#F59E0B",  # amber
    "#EF4444",  # red
    "#8B5CF6",  # purple
    "#EC4899",  # pink
    "#06B6D4",  # cyan
    "#F97316",  # orange
]

BG_COLOR = "#0B1120"
PAPER_COLOR = "#0B1120"
GRID_COLOR = "rgba(148, 163, 184, 0.08)"
TEXT_COLOR = "#E2E8F0"
MUTED_COLOR = "#94A3B8"
CARD_BG = "#151D2E"

_BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(family="DM Sans, Helvetica, Arial, sans-serif", color=TEXT_COLOR, size=13),
    margin=dict(l=60, r=30, t=50, b=80),
    xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
    yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color=MUTED_COLOR),
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="center",
        x=0.5,
    ),
    hoverlabel=dict(
        bgcolor="#1E293B",
        font=dict(size=13, family="DM Sans, monospace"),
        bordercolor="#334155",
    ),
)


def _apply_layout(fig: go.Figure, **overrides) -> go.Figure:
    """Apply consistent dark theme layout."""
    layout = {**_BASE_LAYOUT, **overrides}
    fig.update_layout(**layout)
    return fig


def _dollar_format(val: float) -> str:
    """Format large dollar values."""
    if abs(val) >= 1e6:
        return f"${val/1e6:,.2f}M"
    elif abs(val) >= 1e3:
        return f"${val/1e3:,.1f}K"
    return f"${val:,.0f}"


# ──────────────────────────────────────────────────────────────
# Growth of $1M
# ──────────────────────────────────────────────────────────────


def growth_chart(
    portfolios: dict[str, pd.Series],
    capital: float = 1_000_000,
) -> go.Figure:
    """
    Growth of $X: overlay Active, Passive, ORP, Complete.

    Each series is normalized so first value = capital.
    """
    fig = go.Figure()

    for i, (name, series) in enumerate(portfolios.items()):
        s = series.dropna()
        if s.empty:
            continue
        normalized = s / float(s.iloc[0]) * capital
        color = COLORS[i % len(COLORS)]

        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized.values,
            name=name,
            line=dict(color=color, width=2.5 if i == 0 else 2),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Date: %{x|%b %d, %Y}<br>"
                "Value: %{y:$,.0f}<br>"
                "<extra></extra>"
            ),
        ))

    _apply_layout(
        fig,
        title=dict(
            text=f"Growth of {_dollar_format(capital)}",
            font=dict(size=18, color=TEXT_COLOR),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zeroline=False,
            tickformat="$,.0f",
        ),
        xaxis_title="",
        yaxis_title="Portfolio Value",
        height=500,
    )

    return fig


# ──────────────────────────────────────────────────────────────
# Cumulative outperformance
# ──────────────────────────────────────────────────────────────


def outperformance_chart(
    active: pd.Series,
    passive: pd.Series,
) -> go.Figure:
    """Active − Passive cumulative outperformance."""
    aligned = pd.concat(
        [active.rename("A"), passive.rename("P")], axis=1
    ).dropna()

    diff = aligned["A"] - aligned["P"]
    diff = diff - float(diff.iloc[0])

    positive = diff.copy()
    negative = diff.copy()
    positive[positive < 0] = 0
    negative[negative > 0] = 0

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=diff.index, y=positive.values,
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.15)",
        line=dict(color="#10B981", width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=diff.index, y=negative.values,
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.15)",
        line=dict(color="#EF4444", width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=diff.index,
        y=diff.values,
        name="Outperformance",
        line=dict(
            color="#10B981" if float(diff.iloc[-1]) >= 0 else "#EF4444",
            width=2,
        ),
        hovertemplate=(
            "Date: %{x|%b %d, %Y}<br>"
            "Cumulative: %{y:$,.0f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.add_hline(y=0, line_dash="dot", line_color=MUTED_COLOR, opacity=0.4)

    _apply_layout(
        fig,
        title=dict(
            text="Active vs Passive: Cumulative Outperformance",
            font=dict(size=16, color=TEXT_COLOR),
        ),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, tickformat="$,.0f"),
        showlegend=False,
        height=380,
    )

    return fig


# ──────────────────────────────────────────────────────────────
# Efficient Frontier + CAL
# ──────────────────────────────────────────────────────────────


def efficient_frontier_chart(
    frontier_vols: np.ndarray,
    frontier_returns: np.ndarray,
    orp_vol: float,
    orp_return: float,
    rf: float,
    asset_vols: pd.Series | None = None,
    asset_returns: pd.Series | None = None,
) -> go.Figure:
    """
    Efficient frontier curve, ORP star marker, CAL line,
    and optionally individual asset dots.
    """
    fig = go.Figure()

    # Frontier curve
    fig.add_trace(go.Scatter(
        x=frontier_vols,
        y=frontier_returns,
        mode="lines",
        name="Efficient Frontier",
        line=dict(color="#3B82F6", width=3),
        hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))

    # CAL line
    if orp_vol > 0:
        slope = (orp_return - rf) / orp_vol
        max_vol = max(float(frontier_vols.max()) * 1.3, orp_vol * 1.5)
        x_cal = np.linspace(0, max_vol, 100)
        y_cal = rf + slope * x_cal
        fig.add_trace(go.Scatter(
            x=x_cal,
            y=y_cal,
            mode="lines",
            name="Capital Allocation Line",
            line=dict(color="#F59E0B", width=2, dash="dash"),
            hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        ))

    # Risk-free point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[rf],
        mode="markers",
        name=f"Risk-Free ({rf:.1%})",
        marker=dict(color="#94A3B8", size=10, symbol="diamond"),
        hovertemplate=f"Risk-Free Rate: {rf:.2%}<extra></extra>",
    ))

    # ORP marker
    fig.add_trace(go.Scatter(
        x=[orp_vol],
        y=[orp_return],
        mode="markers",
        name="Optimal Risky Portfolio",
        marker=dict(
            color="#EF4444",
            size=14,
            symbol="star",
            line=dict(width=2, color="#FCA5A5"),
        ),
        hovertemplate=(
            f"<b>ORP</b><br>"
            f"Return: {orp_return:.2%}<br>"
            f"Volatility: {orp_vol:.2%}<br>"
            f"<extra></extra>"
        ),
    ))

    # Individual assets
    if asset_vols is not None and asset_returns is not None:
        fig.add_trace(go.Scatter(
            x=asset_vols.values,
            y=asset_returns.values,
            mode="markers+text",
            name="Individual Assets",
            marker=dict(color="#8B5CF6", size=8, opacity=0.8),
            text=asset_vols.index.tolist(),
            textposition="top center",
            textfont=dict(size=10, color=MUTED_COLOR),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Vol: %{x:.2%}<br>"
                "Return: %{y:.2%}<br>"
                "<extra></extra>"
            ),
        ))

    _apply_layout(
        fig,
        title=dict(
            text="Efficient Frontier & Capital Allocation Line",
            font=dict(size=16, color=TEXT_COLOR),
        ),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, tickformat=".0%", title="Annualized Volatility"),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, tickformat=".0%", title="Annualized Return"),
        height=520,
    )

    return fig


# ──────────────────────────────────────────────────────────────
# Correlation heatmap
# ──────────────────────────────────────────────────────────────


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Interactive correlation matrix with hover values."""
    labels = list(corr.columns)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#EF4444"],
            [0.25, "#7F1D1D"],
            [0.5, "#0B1120"],
            [0.75, "#14532D"],
            [1.0, "#10B981"],
        ],
        zmin=-1,
        zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=11, color=TEXT_COLOR),
        hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="ρ", font=dict(color=MUTED_COLOR)),
            tickfont=dict(color=MUTED_COLOR),
        ),
    ))

    _apply_layout(
        fig,
        title=dict(text="Correlation Matrix", font=dict(size=16, color=TEXT_COLOR)),
        height=500,
        xaxis=dict(gridcolor="rgba(0,0,0,0)", showgrid=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", showgrid=False, autorange="reversed"),
    )

    return fig


# ──────────────────────────────────────────────────────────────
# Drawdown curves
# ──────────────────────────────────────────────────────────────


def drawdown_chart(portfolios: dict[str, pd.Series]) -> go.Figure:
    """Drawdown curves for all portfolios."""
    fig = go.Figure()

    for i, (name, values) in enumerate(portfolios.items()):
        v = values.dropna().astype(float).sort_index()
        if v.empty:
            continue
        dd = v / v.cummax() - 1.0
        color = COLORS[i % len(COLORS)]

        fig.add_trace(go.Scatter(
            x=dd.index,
            y=dd.values,
            name=name,
            fill="tozeroy",
            fillcolor=color.replace(")", ", 0.08)").replace("rgb", "rgba") if "rgb" in color else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
            line=dict(color=color, width=1.5),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Date: %{x|%b %d, %Y}<br>"
                "Drawdown: %{y:.2%}<br>"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_color=MUTED_COLOR, opacity=0.3)

    _apply_layout(
        fig,
        title=dict(text="Portfolio Drawdowns", font=dict(size=16, color=TEXT_COLOR)),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, tickformat=".0%", title="Drawdown"),
        xaxis_title="",
        height=420,
    )

    return fig


# ──────────────────────────────────────────────────────────────
# Weights bar chart
# ──────────────────────────────────────────────────────────────


def weights_bar(
    weights: pd.Series,
    title: str = "Portfolio Weights",
) -> go.Figure:
    """Horizontal bar chart of portfolio weights, sorted by magnitude."""
    w = weights.sort_values()

    bar_colors = [
        "#10B981" if v >= 0 else "#EF4444" for v in w.values
    ]

    fig = go.Figure(go.Bar(
        y=w.index,
        x=w.values,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        hovertemplate="<b>%{y}</b>: %{x:.2%}<extra></extra>",
    ))

    _apply_layout(
        fig,
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=True, zerolinecolor=MUTED_COLOR, tickformat=".0%"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
        height=max(300, len(w) * 40 + 100),
    )

    return fig


# ──────────────────────────────────────────────────────────────
# Complete portfolio donut
# ──────────────────────────────────────────────────────────────


def allocation_donut(
    weights: dict[str, float],
    title: str = "Complete Portfolio Allocation",
) -> go.Figure:
    """Donut chart for portfolio allocation."""
    # Filter tiny positions
    filtered = {k: abs(v) for k, v in weights.items() if abs(v) > 0.005}
    total = sum(filtered.values())
    if total == 0:
        return go.Figure()
    filtered = {k: v / total for k, v in filtered.items()}

    labels = list(filtered.keys())
    values = list(filtered.values())

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=COLORS[:len(labels)], line=dict(color=BG_COLOR, width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color=TEXT_COLOR),
        hovertemplate="<b>%{label}</b><br>Weight: %{percent}<br>Value: %{value:.4f}<extra></extra>",
    ))

    _apply_layout(
        fig,
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        height=450,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
    )

    return fig


# ──────────────────────────────────────────────────────────────
# CAPM scatter
# ──────────────────────────────────────────────────────────────


def capm_scatter(
    ticker: str,
    asset_excess: pd.Series,
    market_excess: pd.Series,
    alpha: float,
    beta: float,
) -> go.Figure:
    """CAPM scatter with regression line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=market_excess.values,
        y=asset_excess.values,
        mode="markers",
        name="Monthly Excess Returns",
        marker=dict(color="#3B82F6", size=6, opacity=0.6),
        hovertemplate="Mkt: %{x:.2%}<br>Asset: %{y:.2%}<extra></extra>",
    ))

    x_line = np.linspace(float(market_excess.min()), float(market_excess.max()), 100)
    y_line = alpha + beta * x_line

    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name=f"α={alpha:.4f}, β={beta:.2f}",
        line=dict(color="#EF4444", width=2),
    ))

    _apply_layout(
        fig,
        title=dict(text=f"{ticker} — CAPM Regression", font=dict(size=15, color=TEXT_COLOR)),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=True, zerolinecolor=MUTED_COLOR, tickformat=".1%", title="Market Excess Return"),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=True, zerolinecolor=MUTED_COLOR, tickformat=".1%", title=f"{ticker} Excess Return"),
        height=400,
    )

    return fig


# ──────────────────────────────────────────────────────────────
# Attribution bar
# ──────────────────────────────────────────────────────────────


def attribution_chart(
    df: pd.DataFrame,
    title: str = "Brinson–Fachler Performance Attribution",
) -> go.Figure:
    """Stacked bar of Allocation, Selection, Interaction effects."""
    if df.empty:
        return go.Figure()

    # Determine label column
    label_col = None
    for c in ["Bucket", "Asset", "Sector", "Name"]:
        if c in df.columns:
            label_col = c
            break
    labels = df[label_col].astype(str) if label_col else df.index.astype(str)

    fig = go.Figure()

    for col_name, color in [
        ("Allocation", "#3B82F6"),
        ("Selection", "#10B981"),
        ("Interaction", "#F59E0B"),
    ]:
        if col_name not in df.columns:
            continue
        fig.add_trace(go.Bar(
            x=labels,
            y=df[col_name].values,
            name=col_name,
            marker=dict(color=color),
            hovertemplate=f"<b>%{{x}}</b><br>{col_name}: %{{y:.4f}}<extra></extra>",
        ))

    fig.add_hline(y=0, line_color=MUTED_COLOR, opacity=0.3)

    _apply_layout(
        fig,
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        barmode="relative",
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, tickformat=".2%", title="Contribution"),
        height=450,
    )

    return fig


# ──────────────────────────────────────────────────────────────
# Rolling metrics
# ──────────────────────────────────────────────────────────────


def rolling_metrics_chart(
    monthly_returns: pd.DataFrame,
    window: int = 12,
    rf_annual: float = 0.04,
) -> go.Figure:
    """Rolling volatility + Sharpe in a 2-panel layout."""
    rolling_vol = monthly_returns.rolling(window).std() * np.sqrt(12)
    rolling_mean = monthly_returns.rolling(window).mean() * 12
    rolling_sharpe = (rolling_mean - rf_annual) / rolling_vol

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["Rolling Annualized Volatility", "Rolling Sharpe Ratio"],
    )

    for i, col in enumerate(rolling_vol.columns):
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol[col],
            name=col, line=dict(color=color, width=1.5),
            legendgroup=col,
            hovertemplate=f"<b>{col}</b><br>Vol: %{{y:.1%}}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe[col],
            name=col, line=dict(color=color, width=1.5),
            legendgroup=col, showlegend=False,
            hovertemplate=f"<b>{col}</b><br>Sharpe: %{{y:.2f}}<extra></extra>",
        ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color=MUTED_COLOR, opacity=0.4, row=2, col=1)

    _apply_layout(
        fig,
        height=600,
        title=dict(text=f"Rolling {window}-Month Risk Analytics", font=dict(size=16, color=TEXT_COLOR)),
    )
    fig.update_yaxes(tickformat=".0%", gridcolor=GRID_COLOR, row=1, col=1)
    fig.update_yaxes(gridcolor=GRID_COLOR, row=2, col=1)
    fig.update_xaxes(gridcolor=GRID_COLOR)

    # Fix subplot title colors
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=13, color=MUTED_COLOR)

    return fig


# ──────────────────────────────────────────────────────────────
# VaR / CVaR histogram
# ──────────────────────────────────────────────────────────────


def return_distribution_chart(
    returns: pd.Series,
    var_95: float | None = None,
    cvar_95: float | None = None,
    title: str = "Daily Return Distribution",
) -> go.Figure:
    """Histogram of daily returns with VaR/CVaR lines."""
    r = returns.dropna()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=r.values,
        nbinsx=60,
        marker=dict(color="#3B82F6", line=dict(width=0.5, color="#1E3A5F")),
        opacity=0.8,
        hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>",
    ))

    if var_95 is not None:
        fig.add_vline(
            x=var_95, line_dash="dash", line_color="#EF4444", line_width=2,
            annotation_text=f"VaR 95%: {var_95:.2%}",
            annotation_font=dict(color="#EF4444", size=11),
            annotation_position="top left",
        )

    if cvar_95 is not None:
        fig.add_vline(
            x=cvar_95, line_dash="dash", line_color="#F59E0B", line_width=2,
            annotation_text=f"CVaR 95%: {cvar_95:.2%}",
            annotation_font=dict(color="#F59E0B", size=11),
            annotation_position="top left",
        )

    _apply_layout(
        fig,
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=True, zerolinecolor=MUTED_COLOR, tickformat=".1%", title="Daily Return"),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, title="Frequency"),
        showlegend=False,
        height=400,
    )

    return fig


# ──────────────────────────────────────────────────────────────
# Factor loadings
# ──────────────────────────────────────────────────────────────


def factor_loadings_chart(
    df: pd.DataFrame,
    model_name: str = "",
) -> go.Figure:
    """Grouped bar chart of factor betas across assets."""
    if df.empty:
        return go.Figure()

    beta_cols = [c for c in df.columns if c.endswith("_coef") and c != "const_coef"]
    if not beta_cols:
        return go.Figure()

    fig = go.Figure()
    assets = df["Asset"].tolist() if "Asset" in df.columns else df.index.tolist()

    for i, col in enumerate(beta_cols):
        factor = col.replace("_coef", "")
        fig.add_trace(go.Bar(
            x=assets,
            y=df[col].values,
            name=factor,
            marker=dict(color=COLORS[i % len(COLORS)]),
            hovertemplate=f"<b>%{{x}}</b><br>{factor}: %{{y:.3f}}<extra></extra>",
        ))

    fig.add_hline(y=0, line_color=MUTED_COLOR, opacity=0.3)

    _apply_layout(
        fig,
        title=dict(
            text=f"Factor Loadings{' — ' + model_name if model_name else ''}",
            font=dict(size=16, color=TEXT_COLOR),
        ),
        barmode="group",
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, title="Beta"),
        height=450,
    )

    return fig


# ──────────────────────────────────────────────────────────────
# Monte Carlo fan chart
# ──────────────────────────────────────────────────────────────


def simulation_fan_chart(
    starting_value: float,
    paths: np.ndarray,
    horizon_days: int,
    method_name: str = "Monte Carlo",
    realized: pd.Series | None = None,
) -> go.Figure:
    """
    Fan chart showing simulation percentile bands.

    Parameters
    ----------
    starting_value : initial portfolio value
    paths : (n_paths, horizon_days) simulation paths
    horizon_days : number of trading days
    method_name : label for the method
    realized : optional historical value series to overlay
    """
    fig = go.Figure()

    # Generate x-axis as months from now
    months = np.arange(horizon_days) / 21.0

    # Percentile bands
    p5 = np.percentile(paths, 5, axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    # P5-P95 band (lightest)
    fig.add_trace(go.Scatter(
        x=np.concatenate([months, months[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.06)",
        line=dict(width=0),
        name="P5–P95",
        hoverinfo="skip",
    ))

    # P10-P90 band
    fig.add_trace(go.Scatter(
        x=np.concatenate([months, months[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.10)",
        line=dict(width=0),
        name="P10–P90",
        hoverinfo="skip",
    ))

    # P25-P75 band (darkest)
    fig.add_trace(go.Scatter(
        x=np.concatenate([months, months[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.18)",
        line=dict(width=0),
        name="P25–P75",
        hoverinfo="skip",
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=months,
        y=p50,
        mode="lines",
        name="Median (P50)",
        line=dict(color="#3B82F6", width=2.5),
        hovertemplate="Month %{x:.0f}<br>Value: %{y:$,.0f}<extra></extra>",
    ))

    # Starting value line
    fig.add_hline(
        y=starting_value,
        line_dash="dot",
        line_color=MUTED_COLOR,
        opacity=0.5,
        annotation_text=f"Start: {_dollar_format(starting_value)}",
        annotation_font=dict(color=MUTED_COLOR, size=10),
    )

    _apply_layout(
        fig,
        title=dict(
            text=f"{method_name} — 3-Year Forward Simulation (500 paths)",
            font=dict(size=16, color=TEXT_COLOR),
        ),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, title="Months Forward"),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, tickformat="$,.0f", title="Portfolio Value"),
        height=480,
    )

    return fig


# ──────────────────────────────────────────────────────────────
# Risk contribution bar
# ──────────────────────────────────────────────────────────────


def risk_contribution_chart(
    risk_pct: pd.Series,
    title: str = "Risk Contribution by Asset",
) -> go.Figure:
    """Horizontal bar chart of percentage risk contribution."""
    r = risk_pct.sort_values()

    fig = go.Figure(go.Bar(
        y=r.index,
        x=r.values,
        orientation="h",
        marker=dict(
            color=[COLORS[i % len(COLORS)] for i in range(len(r))],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
    ))

    _apply_layout(
        fig,
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, ticksuffix="%", title="% of Total Risk"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
        height=max(300, len(r) * 40 + 100),
    )

    return fig


# ──────────────────────────────────────────────────────────────
# Stress test heatmap
# ──────────────────────────────────────────────────────────────


def correlation_regime_chart(corr_df: pd.DataFrame) -> go.Figure:
    """
    Line chart of average pairwise correlation over time.

    Highlights periods where correlation exceeds mean + 1 std (regime stress).
    """
    if corr_df.empty or "AvgCorrelation" not in corr_df.columns:
        return go.Figure()

    avg = corr_df["AvgCorrelation"]
    mean_val = float(avg.mean())
    std_val = float(avg.std())
    upper = mean_val + std_val
    lower = mean_val - std_val

    fig = go.Figure()

    # Band for mean +/- 1 std
    fig.add_trace(go.Scatter(
        x=list(avg.index) + list(avg.index[::-1]),
        y=[upper] * len(avg) + [lower] * len(avg),
        fill="toself",
        fillcolor="rgba(148, 163, 184, 0.08)",
        line=dict(width=0),
        name="Mean +/- 1 Std",
        hoverinfo="skip",
    ))

    # Mean line
    fig.add_hline(
        y=mean_val, line_dash="dot", line_color=MUTED_COLOR, opacity=0.5,
    )

    # Color segments: red when above threshold, blue otherwise
    colors = ["#EF4444" if v > upper else "#3B82F6" for v in avg.values]

    # Main line
    fig.add_trace(go.Scatter(
        x=avg.index,
        y=avg.values,
        mode="lines",
        name="Avg Pairwise Correlation",
        line=dict(color="#3B82F6", width=2),
        hovertemplate="Date: %{x|%b %d, %Y}<br>Avg Corr: %{y:.3f}<extra></extra>",
    ))

    # Overlay red segments where stressed
    stressed = avg.copy()
    stressed[stressed <= upper] = np.nan
    if stressed.notna().any():
        fig.add_trace(go.Scatter(
            x=stressed.index,
            y=stressed.values,
            mode="lines",
            name="High Correlation Regime",
            line=dict(color="#EF4444", width=2.5),
            hovertemplate="Date: %{x|%b %d, %Y}<br>Avg Corr: %{y:.3f}<extra></extra>",
            connectgaps=False,
        ))

    _apply_layout(
        fig,
        title=dict(text="Correlation Regime Detection", font=dict(size=16, color=TEXT_COLOR)),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, title="Avg Pairwise Correlation"),
        xaxis_title="",
        height=400,
    )

    return fig


def dendrogram_chart(
    linkage_matrix: np.ndarray,
    labels: list[str],
) -> go.Figure:
    """
    Render a dendrogram from a scipy linkage matrix using Plotly.

    Uses scipy.cluster.hierarchy.dendrogram for coordinates,
    then draws with go.Scatter.
    """
    from scipy.cluster.hierarchy import dendrogram as scipy_dend

    dend = scipy_dend(linkage_matrix, labels=labels, no_plot=True)

    fig = go.Figure()

    # Draw each U-shaped link
    icoord = dend["icoord"]
    dcoord = dend["dcoord"]

    for i, (xs, ys) in enumerate(zip(icoord, dcoord)):
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    # X-axis labels
    tick_positions = list(range(5, len(labels) * 10 + 1, 10))
    reordered_labels = dend["ivl"]

    _apply_layout(
        fig,
        title=dict(text="HRP Cluster Dendrogram", font=dict(size=16, color=TEXT_COLOR)),
        xaxis=dict(
            tickmode="array",
            tickvals=tick_positions[:len(reordered_labels)],
            ticktext=reordered_labels,
            gridcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, title="Distance"),
        showlegend=False,
        height=400,
    )

    return fig


def weight_drift_chart(drift_df: pd.DataFrame) -> go.Figure:
    """Stacked area chart of portfolio weight drift over time."""
    if drift_df.empty:
        return go.Figure()

    fig = go.Figure()

    for i, col in enumerate(drift_df.columns):
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=drift_df.index,
            y=drift_df[col].values,
            name=col,
            stackgroup="one",
            line=dict(width=0.5, color=color),
            fillcolor=color.replace(")", ", 0.6)").replace("rgb", "rgba")
            if "rgb" in color
            else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.6)",
            hovertemplate=f"<b>{col}</b><br>Date: %{{x|%b %d, %Y}}<br>Weight: %{{y:.2%}}<extra></extra>",
        ))

    _apply_layout(
        fig,
        title=dict(text="Portfolio Weight Drift Over Time", font=dict(size=16, color=TEXT_COLOR)),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, tickformat=".0%", title="Weight"),
        xaxis_title="",
        height=450,
    )

    return fig


def sector_donut_chart(sector_df: pd.DataFrame) -> go.Figure:
    """Donut chart of sector weights."""
    if sector_df.empty:
        return go.Figure()

    labels = sector_df["Sector"].tolist()
    values = sector_df["Weight"].tolist()

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=COLORS[:len(labels)], line=dict(color=BG_COLOR, width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color=TEXT_COLOR),
        hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>",
    ))

    _apply_layout(
        fig,
        title=dict(text="Sector Allocation", font=dict(size=16, color=TEXT_COLOR)),
        height=450,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
    )

    return fig


def factor_tilts_chart(factor_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of factor tilts per asset."""
    if factor_df.empty:
        return go.Figure()

    fig = go.Figure()
    assets = factor_df["Asset"].tolist()

    for i, factor in enumerate(["Beta", "Size", "Momentum", "Quality"]):
        if factor not in factor_df.columns:
            continue
        fig.add_trace(go.Bar(
            x=assets,
            y=factor_df[factor].values,
            name=factor,
            marker=dict(color=COLORS[i % len(COLORS)]),
            hovertemplate=f"<b>%{{x}}</b><br>{factor}: %{{y:.3f}}<extra></extra>",
        ))

    fig.add_hline(y=0, line_color=MUTED_COLOR, opacity=0.3)

    _apply_layout(
        fig,
        title=dict(text="Factor Tilts by Asset", font=dict(size=16, color=TEXT_COLOR)),
        barmode="group",
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, title="Factor Loading"),
        height=450,
    )

    return fig


def income_bar_chart(income_df: pd.DataFrame) -> go.Figure:
    """Bar chart of annual dividend income per holding."""
    if income_df.empty or "AnnualIncome" not in income_df.columns:
        return go.Figure()

    df = income_df.sort_values("AnnualIncome", ascending=True)

    fig = go.Figure(go.Bar(
        y=df["Ticker"],
        x=df["AnnualIncome"],
        orientation="h",
        marker=dict(
            color=[COLORS[i % len(COLORS)] for i in range(len(df))],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b>: $%{x:,.2f}/yr<extra></extra>",
    ))

    _apply_layout(
        fig,
        title=dict(text="Annual Dividend Income by Holding", font=dict(size=16, color=TEXT_COLOR)),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, tickformat="$,.0f", title="Annual Income"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
        height=max(300, len(df) * 40 + 100),
    )

    return fig


def cumulative_income_chart(cum_income: pd.Series) -> go.Figure:
    """Line chart of cumulative dividend income over time."""
    if cum_income.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cum_income.index,
        y=cum_income.values,
        mode="lines",
        name="Cumulative Income",
        line=dict(color="#10B981", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.1)",
        hovertemplate="Date: %{x|%b %d, %Y}<br>Cumulative: $%{y:,.2f}<extra></extra>",
    ))

    _apply_layout(
        fig,
        title=dict(text="Cumulative Dividend Income", font=dict(size=16, color=TEXT_COLOR)),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, tickformat="$,.0f", title="Cumulative Income ($)"),
        xaxis_title="",
        height=400,
    )

    return fig


def stress_test_chart(stress_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart comparing portfolio vs benchmark during stress events."""
    if stress_df.empty:
        return go.Figure()

    # Filter to scenarios with data
    df = stress_df[stress_df["Portfolio"] != "N/A"].copy()
    if df.empty:
        return go.Figure()

    # Parse percentage strings back to floats
    df["Port_f"] = df["Portfolio"].str.rstrip("%").astype(float) / 100
    df["Bench_f"] = df["Benchmark"].str.rstrip("%").astype(float) / 100

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df["Scenario"],
        x=df["Port_f"],
        name="Portfolio",
        orientation="h",
        marker=dict(color="#3B82F6"),
        hovertemplate="<b>%{y}</b><br>Portfolio: %{x:.2%}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        y=df["Scenario"],
        x=df["Bench_f"],
        name="Benchmark",
        orientation="h",
        marker=dict(color="#64748B"),
        hovertemplate="<b>%{y}</b><br>Benchmark: %{x:.2%}<extra></extra>",
    ))

    fig.add_vline(x=0, line_color=MUTED_COLOR, opacity=0.4)

    _apply_layout(
        fig,
        title=dict(text="Stress Test: Historical Scenarios", font=dict(size=16, color=TEXT_COLOR)),
        barmode="group",
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, tickformat=".0%", title="Total Return"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=max(350, len(df) * 55 + 120),
    )

    return fig
