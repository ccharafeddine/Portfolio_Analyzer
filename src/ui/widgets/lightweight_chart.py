"""A TradingView-style candlestick chart, powered by TradingView's own
open-source **Lightweight Charts** library (Apache-2.0), rendered in an embedded
web view.

Design mirrors :class:`PlotlyWidget`: the library JS is vendored under
``assets/vendor`` and loaded from a local file URL (no CDN, works offline); a
stable shell page is loaded once and data is pushed in via ``runJavaScript`` so
there is no reload/flash on symbol or timeframe changes. Feed it an OHLCV frame
via :meth:`set_ohlc`; it renders candlesticks + a volume histogram, a dotted
previous-close line for intraday, and re-tints to the active theme on
:meth:`retheme`.
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from PySide6.QtCore import QUrl
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget

from src.data.market_data import INTRADAY_TIMEFRAMES

from .. import assets, theme

# Referenced by bare filename; the web view's base URL points into the vendor
# dir (below), mirroring how PlotlyWidget loads plotly.min.js.
_LWC_JS = "lightweight-charts.standalone.production.js"


def _rgba(hex_color: str, alpha: float) -> str:
    """A translucent rgba() string from a #RRGGBB color (for volume bars)."""
    h = (hex_color or "").lstrip("#")
    if len(h) != 6:
        return f"rgba(128,128,128,{alpha})"
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


class LightweightChartWidget(QWidget):
    """Candlestick + volume chart for one symbol at one timeframe."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._loaded = False
        self._pending_js: Optional[str] = None
        # Last render inputs, so a theme change can re-tint from stored data.
        self._last: Optional[tuple] = None

        vendor_dir = assets.asset("vendor")
        self._base_url = QUrl.fromLocalFile(vendor_dir + "/")

        bg = QColor(theme.ACTIVE.chart_bg)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.Window, bg)
        self.setPalette(pal)

        self._view = QWebEngineView(self)
        self._view.page().setBackgroundColor(bg)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._view.loadFinished.connect(self._on_loaded)
        self._view.setHtml(self._shell_html(), self._base_url)

    # ── Shell page (loaded once) ──
    def _shell_html(self) -> str:
        bg = theme.ACTIVE.chart_bg
        return f"""<!DOCTYPE html><html><head>
<meta charset='utf-8'/>
<script src='{_LWC_JS}'></script>
<style>
  html,body{{height:100%;margin:0;overflow:hidden;background:{bg}}}
  #c{{width:100%;height:100%}}
</style></head>
<body><div id='c'></div>
<script>
  let chart, candle, vol, prevLine;
  function ensureChart(){{
    if (chart) return;
    chart = LightweightCharts.createChart(document.getElementById('c'), {{autoSize:true}});
    candle = chart.addCandlestickSeries({{borderVisible:false}});
    vol = chart.addHistogramSeries({{priceFormat:{{type:'volume'}}, priceScaleId:''}});
    vol.priceScale().applyOptions({{scaleMargins:{{top:0.82, bottom:0}}}});
  }}
  function applyData(p){{
    ensureChart();
    const o = p.opts;
    chart.applyOptions({{
      layout:{{background:{{type:'solid', color:o.bg}}, textColor:o.text}},
      grid:{{vertLines:{{color:o.grid}}, horzLines:{{color:o.grid}}}},
      rightPriceScale:{{borderColor:o.grid}},
      timeScale:{{borderColor:o.grid, timeVisible:o.intraday, secondsVisible:false}},
      crosshair:{{mode: LightweightCharts.CrosshairMode.Normal}},
    }});
    candle.applyOptions({{upColor:o.up, downColor:o.down,
      wickUpColor:o.up, wickDownColor:o.down, borderVisible:false}});
    candle.setData(p.candles);
    vol.setData(p.volumes);
    if (prevLine) {{ candle.removePriceLine(prevLine); prevLine = null; }}
    if (o.prevClose != null) {{
      prevLine = candle.createPriceLine({{price:o.prevClose, color:o.muted,
        lineWidth:1, lineStyle:1, axisLabelVisible:true, title:'prev'}});
    }}
    chart.timeScale().fitContent();
  }}
  function clearData(){{ if (candle) candle.setData([]); if (vol) vol.setData([]); }}
</script></body></html>"""

    def _on_loaded(self, _ok: bool) -> None:
        self._loaded = True
        if self._pending_js is not None:
            self._view.page().runJavaScript(self._pending_js)
            self._pending_js = None

    def _run(self, js: str) -> None:
        if self._loaded:
            self._view.page().runJavaScript(js)
        else:
            self._pending_js = js

    # ── Public API ──
    def set_ohlc(self, ticker: str, df, timeframe: str, prev_close=None) -> None:
        """Render ``df`` (a yfinance OHLCV frame) as candlesticks + volume. Passing
        ``df=None`` clears the chart. Stores inputs so :meth:`retheme` can re-tint."""
        self._last = (ticker, df, timeframe, prev_close)
        payload = self._build_payload(df, timeframe, prev_close)
        if payload is None:
            self._run("clearData()")
            return
        self._run(f"applyData({json.dumps(payload)})")

    def clear(self) -> None:
        self._last = None
        self._run("clearData()")

    def retheme(self) -> None:
        if self._last is not None and self._last[1] is not None:
            ticker, df, timeframe, prev_close = self._last
            self.set_ohlc(ticker, df, timeframe, prev_close)

    # ── Data conversion ──
    def _build_payload(self, df, timeframe: str, prev_close) -> Optional[dict]:
        if df is None or getattr(df, "empty", True):
            return None
        cols = {c.lower(): c for c in df.columns}
        try:
            o_c, h_c, l_c, c_c = cols["open"], cols["high"], cols["low"], cols["close"]
        except KeyError:
            return None
        v_c = cols.get("volume")
        intraday = timeframe in INTRADAY_TIMEFRAMES

        t = theme.ACTIVE
        up, down = t.green, t.red
        candles, volumes = [], []
        for ts, row in df.iterrows():
            time_val = self._time_value(ts, intraday)
            if time_val is None:
                continue
            try:
                o, h, low, c = (float(row[o_c]), float(row[h_c]),
                                float(row[l_c]), float(row[c_c]))
            except (TypeError, ValueError):
                continue
            if any(x != x for x in (o, h, low, c)):  # skip NaN rows
                continue
            candles.append({"time": time_val, "open": o, "high": h, "low": low, "close": c})
            if v_c is not None:
                try:
                    vv = float(row[v_c])
                except (TypeError, ValueError):
                    vv = 0.0
                if vv == vv and vv > 0:
                    color = _rgba(up if c >= o else down, 0.5)
                    volumes.append({"time": time_val, "value": vv, "color": color})
        if not candles:
            return None

        opts = {
            "bg": t.chart_bg, "grid": t.chart_grid, "text": t.text,
            "muted": t.text_muted, "up": up, "down": down,
            "intraday": bool(intraday),
            "prevClose": float(prev_close) if (intraday and isinstance(prev_close, (int, float))) else None,
        }
        return {"candles": candles, "volumes": volumes, "opts": opts}

    @staticmethod
    def _time_value(ts, intraday: bool):
        """Lightweight Charts time: UNIX seconds for intraday, 'YYYY-MM-DD' for
        daily+ bars. Returns None if the index value can't be parsed.

        Lightweight Charts renders timestamps in UTC, so for intraday we feed it
        the *exchange wall-clock* time reinterpreted as UTC — otherwise a 09:30 ET
        bar would display as 13:30. Daily bars use a plain date string."""
        try:
            stamp = pd.Timestamp(ts)
        except Exception:
            return None
        if intraday:
            try:
                naive = stamp.replace(tzinfo=None)  # keep wall-clock, drop tz
                return int(pd.Timestamp(naive, tz="UTC").timestamp())
            except Exception:
                return None
        return stamp.strftime("%Y-%m-%d")
