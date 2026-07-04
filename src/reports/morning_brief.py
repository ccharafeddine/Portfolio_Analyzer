"""Morning Brief — a light, market-open snapshot of one portfolio.

Unlike the full analytical report (backtest / risk / optimization), this is a fast
"here's your portfolio this morning" briefing built from live delayed quotes plus
the upcoming-events calendar and latest news — no pipeline run. Qt-free and pure:
:func:`build_morning_brief` takes already-fetched data and returns a self-contained
HTML document, an email subject, and a one-line summary (for the tray notification).

The HTML is rendered through an autoescaping Jinja2 environment, so untrusted
values (tickers, news titles) can't inject markup into the emailed report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from jinja2 import Environment, select_autoescape

_UP = "#10B981"
_DOWN = "#EF4444"
_MUTED = "#64748B"

_env = Environment(autoescape=select_autoescape(default=True))


def _num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and v == v else None


def _pct(v: Optional[float]) -> str:
    return f"{v * 100:+.2f}%" if v is not None else "—"


def _price(v: Optional[float]) -> str:
    return f"{v:,.2f}" if v is not None else "—"


def _money(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{'-' if v < 0 else ''}${abs(v):,.0f}"


def _rel_time(dt: Optional[datetime]) -> str:
    if not isinstance(dt, datetime):
        return ""
    secs = (datetime.now(timezone.utc) - dt).total_seconds()
    if secs < 0:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return dt.strftime("%b %d")


def _event_when(iso: str, today: date) -> str:
    try:
        d = datetime.fromisoformat(iso).date()
    except Exception:
        return iso
    days = (d - today).days
    return "today" if days == 0 else ("tomorrow" if days == 1 else f"in {days}d")


@dataclass
class BriefData:
    name: str
    as_of: date
    total_value: float
    day_pct: Optional[float]
    day_pnl: Optional[float]
    holdings: list = field(default_factory=list)  # {ticker,last,chg_pct,weight,up}
    events: list = field(default_factory=list)     # {date,ticker,kind,detail,when}
    news: list = field(default_factory=list)       # {title,url,publisher,when}


def compute_brief(config, quotes: dict, events=None, news=None,
                  as_of: Optional[date] = None, name: str = "Portfolio") -> BriefData:
    """Reduce a config + fetched data into the numbers the brief shows.

    Weighted day change is over holdings that have both a weight and a quote; day
    P&L applies it to the *invested* amount (``capital - cash``), since cash has no
    day move and ``capital`` is the total account value."""
    weights = dict(getattr(config, "weights", {}) or {})
    tickers = list(getattr(config, "tickers", []) or []) or list(weights)
    capital = float(getattr(config, "capital", 0.0) or 0.0)
    cash = float(getattr(config, "cash", 0.0) or 0.0)
    quotes = quotes or {}

    wsum = wpct = 0.0
    for sym, w in weights.items():
        pct = _num(getattr(quotes.get(sym), "change_pct", None))
        if pct is not None:
            wsum += w
            wpct += w * pct
    day_pct = (wpct / wsum) if wsum > 0 else None
    invested = max(capital - cash, 0.0)
    day_pnl = (invested * day_pct) if (day_pct is not None and invested) else None

    holdings = []
    for sym in tickers:
        q = quotes.get(sym)
        pct = _num(getattr(q, "change_pct", None))
        holdings.append({
            "ticker": sym,
            "last": _price(_num(getattr(q, "last", None))),
            "chg_pct": _pct(pct),
            "weight": (f"{weights[sym] * 100:.1f}%" if sym in weights else "—"),
            "up": (pct is None) or (pct >= 0),
            "neutral": pct is None,
        })

    today = as_of or datetime.now().date()
    ev = []
    for e in (events or [])[:8]:
        ev.append({
            "date": getattr(e, "date", ""),
            "ticker": getattr(e, "ticker", ""),
            "kind": getattr(e, "kind", ""),
            "detail": getattr(e, "detail", "") or "",
            "when": _event_when(getattr(e, "date", ""), today),
        })

    nw = []
    for n in (news or [])[:6]:
        url = (getattr(n, "url", "") or "").strip()
        nw.append({
            "title": getattr(n, "title", "") or "",
            "url": url if url.lower().startswith(("http://", "https://")) else "",
            "publisher": getattr(n, "publisher", "") or "",
            "when": _rel_time(getattr(n, "published", None)),
        })

    return BriefData(
        name=name, as_of=today, total_value=capital,
        day_pct=day_pct, day_pnl=day_pnl,
        holdings=holdings, events=ev, news=nw,
    )


_TEMPLATE = _env.from_string("""\
<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="script-src 'none'; object-src 'none'; base-uri 'none'">
<title>Morning Brief — {{ d.name }}</title>
<style>
  body{font-family:'DM Sans',Segoe UI,Helvetica,Arial,sans-serif;background:#0B1120;color:#E2E8F0;margin:0;padding:24px}
  .wrap{max-width:680px;margin:0 auto}
  h1{font-size:22px;margin:0 0 2px}
  .date{color:#94A3B8;font-size:13px;margin-bottom:18px}
  .cards{display:flex;gap:12px;margin-bottom:22px;flex-wrap:wrap}
  .card{background:#151D2E;border:1px solid #1E293B;border-radius:12px;padding:14px 18px;flex:1;min-width:150px}
  .label{color:#64748B;font-size:11px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}
  .value{font-size:22px;font-weight:700}
  h2{font-size:14px;text-transform:uppercase;letter-spacing:.06em;color:#94A3B8;margin:24px 0 8px;border-bottom:1px solid #1E293B;padding-bottom:6px}
  table{width:100%;border-collapse:collapse;font-size:14px}
  th{text-align:left;color:#64748B;font-weight:600;padding:6px 8px;font-size:12px}
  td{padding:6px 8px;border-top:1px solid #1E293B}
  td.num{text-align:right;font-variant-numeric:tabular-nums}
  .news a{color:#E2E8F0;text-decoration:none;font-weight:600}
  .news .meta{color:#64748B;font-size:12px}
  .news li{margin-bottom:10px;list-style:none}
  ul{padding:0;margin:0}
  .foot{color:#64748B;font-size:11px;margin-top:26px;border-top:1px solid #1E293B;padding-top:12px}
</style></head>
<body><div class="wrap">
  <h1>Morning Brief — {{ d.name }}</h1>
  <div class="date">{{ d.as_of.strftime('%A, %B %d, %Y') }}</div>

  <div class="cards">
    <div class="card"><div class="label">Total Value</div><div class="value">${{ '{:,.0f}'.format(d.total_value) }}</div></div>
    <div class="card"><div class="label">Day Change</div><div class="value" style="color:{{ up if (d.day_pct or 0) >= 0 else down }}">{{ pct(d.day_pct) }}</div></div>
    <div class="card"><div class="label">Day P&amp;L</div><div class="value" style="color:{{ up if (d.day_pnl or 0) >= 0 else down }}">{{ money(d.day_pnl) }}</div></div>
  </div>

  <h2>Holdings</h2>
  <table>
    <tr><th>Ticker</th><th style="text-align:right">Last</th><th style="text-align:right">Day</th><th style="text-align:right">Weight</th></tr>
    {% for h in d.holdings %}
    <tr>
      <td>{{ h.ticker }}</td>
      <td class="num">{{ h.last }}</td>
      <td class="num" style="color:{{ muted if h.neutral else (up if h.up else down) }}">{{ h.chg_pct }}</td>
      <td class="num">{{ h.weight }}</td>
    </tr>
    {% endfor %}
  </table>

  {% if d.events %}
  <h2>Today &amp; Upcoming</h2>
  <table>
    {% for e in d.events %}
    <tr><td>{{ e.ticker }}</td><td>{{ e.kind }}</td><td>{{ e.detail }}</td><td class="num" style="color:{{ muted }}">{{ e.when }}</td></tr>
    {% endfor %}
  </table>
  {% endif %}

  {% if d.news %}
  <h2>Latest News</h2>
  <ul class="news">
    {% for n in d.news %}
    <li>
      {% if n.url %}<a href="{{ n.url }}">{{ n.title }}</a>{% else %}<span style="font-weight:600">{{ n.title }}</span>{% endif %}
      <div class="meta">{{ n.publisher }}{% if n.when %} · {{ n.when }}{% endif %}</div>
    </li>
    {% endfor %}
  </ul>
  {% endif %}

  <div class="foot">
    Generated by Portfolio Analyzer. Delayed quotes for informational and educational purposes only —
    not investment advice. Prices may be 15+ minutes delayed or reflect the prior close.
  </div>
</div></body></html>""")


def render_brief_html(d: BriefData) -> str:
    return _TEMPLATE.render(d=d, pct=_pct, money=_money, up=_UP, down=_DOWN, muted=_MUTED)


def build_morning_brief(config, quotes: dict, events=None, news=None,
                        as_of: Optional[date] = None, name: str = "Portfolio") -> dict:
    """Return ``{html, subject, summary, data}`` for one portfolio's morning brief."""
    d = compute_brief(config, quotes, events=events, news=news, as_of=as_of, name=name)
    subject = f"Morning Brief · {name} · {_pct(d.day_pct)} ({d.as_of.strftime('%b %d')})"
    summary = f"{name}: {_pct(d.day_pct)} day change · {_money(d.day_pnl)} P&L"
    return {"html": render_brief_html(d), "subject": subject, "summary": summary, "data": d}
