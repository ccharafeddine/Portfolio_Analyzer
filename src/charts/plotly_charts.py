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
HOVER_BORDER = "#334155"
# Plotly base template. Flipped to "plotly_white" for light themes so chart
# internals (axes, ticks, spikelines, colorbars) render dark-on-light rather
# than the default dark-on-dark. ``apply_palette(light=...)`` sets this.
TEMPLATE = "plotly_dark"
IS_LIGHT = False

def _make_base_layout() -> dict:
    """Build the shared layout dict from the current module palette."""
    return dict(
        template=TEMPLATE,
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
            bgcolor=CARD_BG,
            font=dict(size=13, family="DM Sans, monospace", color=TEXT_COLOR),
            bordercolor=HOVER_BORDER,
        ),
    )


_BASE_LAYOUT = _make_base_layout()


def apply_palette(
    bg: str,
    paper: str,
    grid: str,
    text: str,
    muted: str,
    card: str | None = None,
    border: str | None = None,
    series: "list[str] | tuple[str, ...] | None" = None,
    light: bool = False,
) -> None:
    """Retint the charts to match the active UI theme.

    Reassigns the module palette and rebuilds the shared base layout. Charts
    created after this call use the new colors (existing figures are not
    mutated — re-render them). Defaults are preserved if never called.

    ``light`` flips the Plotly base template (``plotly_white`` vs
    ``plotly_dark``) so the chart internals go genuinely light, not just the
    background fill. ``series`` overrides the categorical color cycle
    (``COLORS``). ``card``/``border`` retint the hover tooltip so it matches
    the theme's surfaces instead of staying dark on a light chart.
    """
    global BG_COLOR, PAPER_COLOR, GRID_COLOR, TEXT_COLOR, MUTED_COLOR
    global CARD_BG, HOVER_BORDER, IS_LIGHT, TEMPLATE, COLORS, _BASE_LAYOUT
    BG_COLOR = bg
    PAPER_COLOR = paper
    GRID_COLOR = grid
    TEXT_COLOR = text
    MUTED_COLOR = muted
    IS_LIGHT = light
    TEMPLATE = "plotly_white" if light else "plotly_dark"
    if series:
        COLORS = list(series)
    if card is not None:
        CARD_BG = card
    if border is not None:
        HOVER_BORDER = border
    _BASE_LAYOUT = _make_base_layout()


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
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zeroline=False,
            tickformat="$,.0f",
            title=dict(text="Cumulative $ vs benchmark (starts at $0)"),
        ),
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
        margin=dict(l=60, r=30, t=50, b=70),
        legend=dict(
            x=0.98, y=0.98, xanchor="right", yanchor="top",
            bgcolor="rgba(11,17,32,0.8)", bordercolor="#1E293B", borderwidth=1,
            orientation="v",
        ),
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
            [0.5, BG_COLOR],  # zero correlation blends into the chart background
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
        legend=dict(
            x=0.02, y=0.98, xanchor="left", yanchor="top",
            bgcolor="rgba(11,17,32,0.8)", bordercolor="#1E293B", borderwidth=1,
        ),
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
    fig.update_layout(legend=dict(
        x=0.5, y=1.02, xanchor="center", yanchor="bottom",
        orientation="h", bgcolor="rgba(0,0,0,0)",
    ))
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


def contribution_bar_chart(
    totals: pd.Series, title: str = "Contribution to Active Return by Holding"
) -> go.Figure:
    """Per-holding total contribution to active return (green add / red drag)."""
    if totals is None or len(totals) == 0:
        return go.Figure()
    s = totals.sort_values(ascending=False)
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in s.values]
    fig = go.Figure(go.Bar(
        x=[str(i) for i in s.index],
        y=(s.values * 100).tolist(),
        marker=dict(color=colors),
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        cliponaxis=False,
    ))
    fig.add_hline(y=0, line_color=MUTED_COLOR, opacity=0.4)
    _apply_layout(
        fig,
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        yaxis=dict(title="Contribution to active return (%)", gridcolor=GRID_COLOR,
                   zeroline=False),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=360,
    )
    return fig


def attribution_timeseries_chart(contrib: pd.DataFrame) -> go.Figure:
    """Cumulative contribution of each holding to active return over time (+ total)."""
    if contrib is None or contrib.empty:
        return go.Figure()
    cum = contrib.cumsum()
    fig = go.Figure()
    for i, col in enumerate(cum.columns):
        fig.add_trace(go.Scatter(
            x=cum.index, y=(cum[col] * 100).tolist(), name=str(col), mode="lines",
            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            hovertemplate=f"{col}: %{{y:.2f}}%<extra></extra>",
        ))
    total = (cum.sum(axis=1) * 100).tolist()
    fig.add_trace(go.Scatter(
        x=cum.index, y=total, name="Total active", mode="lines",
        line=dict(color=TEXT_COLOR, width=2.6),
        hovertemplate="Total: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=MUTED_COLOR, opacity=0.3)
    _apply_layout(
        fig,
        title=dict(text="Cumulative Contribution to Active Return",
                   font=dict(size=16, color=TEXT_COLOR)),
        yaxis=dict(title="Cumulative contribution (%)", gridcolor=GRID_COLOR, zeroline=False),
        xaxis=dict(gridcolor=GRID_COLOR),
        height=400,
    )
    return fig


def trade_chart(df: pd.DataFrame, title: str = "Trades to Rebalance") -> go.Figure:
    """Per-ticker buy (green) / sell (red) dollar trades."""
    if df is None or df.empty:
        return go.Figure()
    d = df[df["Action"] != "Hold"].copy()
    if d.empty:
        return go.Figure()
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in d["TradeValue"]]
    fig = go.Figure(go.Bar(
        x=d["Ticker"].tolist(),
        y=d["TradeValue"].tolist(),
        marker=dict(color=colors),
        hovertemplate="<b>%{x}</b><br>%{y:$,.0f}<extra></extra>",
        cliponaxis=False,
    ))
    fig.add_hline(y=0, line_color=MUTED_COLOR, opacity=0.4)
    _apply_layout(
        fig,
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Buy (+) / Sell (−) $", gridcolor=GRID_COLOR, zeroline=False),
        height=360,
    )
    return fig


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
        legend=dict(
            x=0.02, y=0.98, xanchor="left", yanchor="top",
            bgcolor="rgba(11,17,32,0.8)", bordercolor="#1E293B", borderwidth=1,
        ),
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
        legend=dict(
            x=0.98, y=0.98, xanchor="right", yanchor="top",
            bgcolor="rgba(11,17,32,0.8)", bordercolor="#1E293B", borderwidth=1,
            orientation="v",
        ),
    )

    return fig


# ──────────────────────────────────────────────────────────────
# Performance suite (Phase 2)
# ──────────────────────────────────────────────────────────────


def capture_chart(cap: dict) -> go.Figure:
    """Up vs down capture ratios (100% = matches the benchmark)."""
    up = cap.get("up_capture")
    down = cap.get("down_capture")
    labels = ["Up Capture", "Down Capture"]
    vals = [
        (up * 100) if up is not None and not np.isnan(up) else 0.0,
        (down * 100) if down is not None and not np.isnan(down) else 0.0,
    ]
    # Good = capture more upside (green) and less downside (green when < 100).
    colors = [
        "#10B981" if vals[0] >= 100 else "#F59E0B",
        "#10B981" if vals[1] <= 100 else "#EF4444",
    ]
    fig = go.Figure(
        go.Bar(
            x=labels, y=vals, marker_color=colors,
            text=[f"{v:.0f}%" for v in vals], textposition="outside",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(y=100, line_dash="dot", line_color=MUTED_COLOR, opacity=0.6)
    _apply_layout(
        fig,
        title=dict(text="Up / Down Capture vs Benchmark", font=dict(size=16, color=TEXT_COLOR)),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, ticksuffix="%", title="% of benchmark move"),
        showlegend=False,
        height=340,
    )
    return fig


def rolling_alpha_beta_chart(df: pd.DataFrame) -> go.Figure:
    """Two-panel rolling annualized alpha (top) and beta (bottom)."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=("Rolling Alpha (annualized)", "Rolling Beta"),
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Alpha"], name="Alpha",
            line=dict(color=COLORS[0], width=2),
            hovertemplate="%{x|%b %Y}<br>Alpha: %{y:.2%}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED_COLOR, opacity=0.5, row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Beta"], name="Beta",
            line=dict(color=COLORS[6], width=2),
            hovertemplate="%{x|%b %Y}<br>Beta: %{y:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=1, line_dash="dot", line_color=MUTED_COLOR, opacity=0.5, row=2, col=1)
    _apply_layout(fig, showlegend=False, height=460)
    fig.update_yaxes(tickformat=".1%", gridcolor=GRID_COLOR, row=1, col=1)
    fig.update_yaxes(gridcolor=GRID_COLOR, row=2, col=1)
    return fig


def coverage_timeline_chart(coverage: dict, end_date) -> go.Figure:
    """Per-asset data availability: when each holding entered the backtest."""
    end = pd.Timestamp(end_date)
    tickers = sorted(coverage.keys(), key=lambda t: coverage[t])
    fig = go.Figure()
    for i, t in enumerate(tickers):
        start = pd.Timestamp(coverage[t])
        fig.add_trace(
            go.Scatter(
                x=[start, end], y=[t, t], mode="lines",
                line=dict(color=COLORS[i % len(COLORS)], width=10),
                hovertemplate=f"{t}<br>available from {start:%b %d, %Y}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[start], y=[t], mode="markers",
                marker=dict(color="#E8EEF6", size=9, line=dict(color=COLORS[i % len(COLORS)], width=2)),
                hoverinfo="skip", showlegend=False,
            )
        )
    _apply_layout(
        fig,
        title=dict(text="Data Coverage by Holding", font=dict(size=16, color=TEXT_COLOR)),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, automargin=True),
        showlegend=False,
        height=max(220, 60 + 34 * len(tickers)),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Macro / rates (News & Macro tab)
# ──────────────────────────────────────────────────────────────

def treasury_curve_chart(curve: dict) -> go.Figure:
    """Yield vs tenor for the current Treasury curve. ``curve`` maps tenor
    labels (e.g. '3M', '2Y', '10Y', '30Y') to yields in percent."""
    order = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    tenors = [t for t in order if t in curve] or list(curve.keys())
    yields = [curve[t] for t in tenors]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tenors, y=yields, mode="lines+markers",
        line=dict(color=COLORS[0], width=2.5),
        marker=dict(size=8, color=COLORS[0]),
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    _apply_layout(
        fig,
        title=dict(text="U.S. Treasury Yield Curve", font=dict(size=16, color=TEXT_COLOR)),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, title="Maturity"),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, title="Yield (%)", ticksuffix="%"),
        showlegend=False,
    )
    return fig


def metric_bar_chart(names, values, title: str, pct: bool = False,
                     suffix: str = "") -> go.Figure:
    """A simple vertical bar comparing one metric across holdings (skips None)."""
    xs, ys = [], []
    for n, v in zip(names, values):
        if v is None:
            continue
        xs.append(n)
        ys.append(v * 100 if pct else v)
    text = [f"{y:.1f}%" if pct else (f"{y:,.2f}{suffix}") for y in ys]
    colors = [COLORS[i % len(COLORS)] for i in range(len(xs))]
    fig = go.Figure(go.Bar(
        x=xs, y=ys, marker_color=colors, text=text, textposition="outside",
        cliponaxis=False, hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    # Headroom so the outside value labels are never clipped at the top/bottom.
    hi = max(ys) if ys else 1.0
    lo = min(ys) if ys else 0.0
    y_top = hi * 1.20 if hi > 0 else hi * 0.80
    y_bot = lo * 1.20 if lo < 0 else 0.0
    _apply_layout(
        fig,
        title=dict(text=title, font=dict(size=15, color=TEXT_COLOR)),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=True,
                   ticksuffix=("%" if pct else ""), range=[y_bot, y_top]),
        showlegend=False,
    )
    return fig


def statement_trend_chart(periods, series: dict, title: str) -> go.Figure:
    """Grouped bars of statement line items over periods (values in USD billions).
    ``series`` maps a label to values aligned with ``periods`` (newest-first)."""
    order = list(reversed([str(p) for p in periods]))
    fig = go.Figure()
    for i, (label, vals) in enumerate(series.items()):
        y = [(v / 1e9 if v is not None else None) for v in reversed(vals)]
        fig.add_trace(go.Bar(
            name=label, x=order, y=y, marker_color=COLORS[i % len(COLORS)],
            hovertemplate="%{x}: $%{y:.1f}B<extra>" + label + "</extra>",
        ))
    _apply_layout(
        fig,
        title=dict(text=title, font=dict(size=15, color=TEXT_COLOR)),
        barmode="group",
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=True, title="USD billions", tickprefix=""),
    )
    return fig


def rate_history_chart(series: dict, title: str = "Key Rates") -> go.Figure:
    """Multi-line history of a few macro series. ``series`` maps a display
    label to a pandas Series indexed by date (values in percent)."""
    fig = go.Figure()
    for i, (label, s) in enumerate(series.items()):
        if s is None or len(s) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=list(s.index), y=list(s.values), mode="lines",
            name=label, line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>" + label + "</extra>",
        ))
    _apply_layout(
        fig,
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, title="Percent", ticksuffix="%"),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Live Market Watch (Tier 2): intraday chart + holdings treemap
# ──────────────────────────────────────────────────────────────

_UP = "#10B981"
_DOWN = "#EF4444"


def intraday_chart(df, ticker: str = "", prev_close=None):
    """Today's 1-minute close line for one ticker, colored by direction vs. the
    previous close, with a dotted reference line at the prior close. ``df`` is a
    yfinance OHLCV frame; returns ``None`` when there's nothing to draw."""
    if df is None or getattr(df, "empty", True):
        return None
    close = df["Close"] if "Close" in getattr(df, "columns", []) else df.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        return None

    base = float(prev_close) if prev_close not in (None, 0) else float(close.iloc[0])
    last = float(close.iloc[-1])
    up = last >= base
    color = _UP if up else _DOWN
    fill = "rgba(16,185,129,0.10)" if up else "rgba(239,68,68,0.10)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(close.index), y=list(close.values), mode="lines",
        line=dict(color=color, width=2), fill="tozeroy", fillcolor=fill,
        name=ticker or "Price",
        hovertemplate="%{x|%H:%M}<br>%{y:,.2f}<extra></extra>",
    ))
    # Tight y-range so the fill reads as a band, not a wedge to zero.
    lo = float(min(close.min(), base))
    hi = float(max(close.max(), base))
    pad = (hi - lo) * 0.12 or (abs(hi) * 0.01 or 1.0)
    fig.add_hline(y=base, line=dict(color=MUTED_COLOR, width=1, dash="dot"))
    _apply_layout(
        fig,
        title=dict(text=f"{ticker} · Intraday (1m)".strip(" ·"),
                   font=dict(size=15, color=TEXT_COLOR)),
        showlegend=False,
        margin=dict(l=54, r=20, t=44, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, range=[lo - pad, hi + pad]),
    )
    return fig


def _tile_color(change_pct) -> str:
    """A green/red tile shade whose intensity scales with the day move
    (saturating at ±3%). Neutral card color when the change is unknown."""
    if change_pct is None or change_pct != change_pct:
        return CARD_BG
    mag = min(1.0, abs(float(change_pct)) / 0.03)
    alpha = 0.30 + 0.55 * mag
    rgb = "16,185,129" if change_pct >= 0 else "239,68,68"
    return f"rgba({rgb},{alpha:.3f})"


def holdings_treemap(tickers, weights: dict, changes: dict):
    """A treemap of holdings sized by portfolio weight and shaded by day change
    %. ``weights``/``changes`` are ``{ticker: value}``; returns ``None`` when no
    holding has a positive weight."""
    labels, values, colors, text = [], [], [], []
    for t in (tickers or []):
        w = weights.get(t)
        if w is None or w <= 0:
            continue
        c = changes.get(t)
        labels.append(t)
        values.append(float(w))
        colors.append(_tile_color(c))
        text.append(f"<b>{t}</b><br>{c * 100:+.2f}%" if isinstance(c, (int, float)) and c == c else f"<b>{t}</b>")
    if not labels:
        return None

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=[""] * len(labels),
        values=values,
        text=text,
        textinfo="text",
        marker=dict(colors=colors, line=dict(color=BG_COLOR, width=2)),
        hovertemplate="%{label}<br>Weight %{value:.1%}<extra></extra>",
        tiling=dict(pad=2),
    ))
    _apply_layout(
        fig,
        title=dict(text="Holdings · day change", font=dict(size=15, color=TEXT_COLOR)),
        margin=dict(l=8, r=8, t=44, b=8),
    )
    return fig
