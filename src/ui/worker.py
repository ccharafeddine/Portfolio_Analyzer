"""Background worker that runs the analysis pipeline off the UI thread.

Uses the ``moveToThread`` pattern (not QThread subclassing). The pipeline's
existing ``progress: Callable[[str, float], None]`` contract maps directly onto a
Qt signal, so no pipeline changes are needed. All heavy work (network fetch +
compute) happens in the worker thread; results are delivered back to the GUI
thread via a queued ``finished`` signal.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot


class AnalysisWorker(QObject):
    progress = Signal(str, float)  # step label, fraction 0..1
    finished = Signal(object)      # AnalysisResults
    failed = Signal(str)           # error message

    def __init__(self, config) -> None:
        super().__init__()
        self._config = config

    @Slot()
    def run(self) -> None:
        try:
            from src.pipeline import AnalysisPipeline
            from . import paths

            pipeline = AnalysisPipeline(
                self._config, output_dir=str(paths.outputs_dir())
            )
            results = pipeline.run(progress=self._emit_progress)
            self.finished.emit(results)
        except Exception as e:  # config/network-level failure
            self.failed.emit(str(e))

    def _emit_progress(self, label: str, frac: float) -> None:
        self.progress.emit(str(label), float(frac))


class NewsWorker(QObject):
    """Fetches per-holding news + an upcoming-events calendar off the UI thread.
    Emits ``done((list[NewsItem], list[CalendarEvent]))``."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, tickers, av_key=None) -> None:
        super().__init__()
        self._tickers = list(tickers or [])
        self._av_key = av_key

    @Slot()
    def run(self) -> None:
        try:
            from src.data import market_data

            news = market_data.fetch_ticker_news(self._tickers, av_key=self._av_key)
            events = market_data.fetch_calendar(self._tickers)
            self.done.emit((news, events))
        except Exception as e:
            self.failed.emit(str(e))


class MacroWorker(QObject):
    """Fetches macro/rates (FRED) off the UI thread. Emits ``done(MacroData|None)``."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fred_key) -> None:
        super().__init__()
        self._fred_key = fred_key

    @Slot()
    def run(self) -> None:
        try:
            from src.data import market_data

            self.done.emit(market_data.fetch_macro(self._fred_key))
        except Exception as e:
            self.failed.emit(str(e))


class FundamentalsWorker(QObject):
    """Fetches per-holding fundamentals off the UI thread. Emits ``done(list)``."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, tickers, fmp_key=None) -> None:
        super().__init__()
        self._tickers = list(tickers or [])
        self._fmp_key = fmp_key

    @Slot()
    def run(self) -> None:
        try:
            from src.data import fundamentals

            self.done.emit(
                fundamentals.fetch_fundamentals(self._tickers, fmp_key=self._fmp_key)
            )
        except Exception as e:
            self.failed.emit(str(e))


class ReportGenWorker(QObject):
    """Generates reports for a list of (name, PortfolioConfig) off the UI thread."""

    progress = Signal(str, float)
    done = Signal(object)   # list[str] of written file paths
    failed = Signal(str)

    def __init__(self, named_configs, out_dir, formats) -> None:
        super().__init__()
        self._named_configs = named_configs
        self._out_dir = out_dir
        self._formats = tuple(formats)

    @Slot()
    def run(self) -> None:
        try:
            from src.reports.generate import generate_report

            written: list[str] = []
            n = max(1, len(self._named_configs))
            for i, (name, cfg) in enumerate(self._named_configs):
                self.progress.emit(f"Report: {name}…", i / n)
                try:
                    for w in generate_report(cfg, self._out_dir,
                                             formats=self._formats, name=name):
                        written.append(str(w))
                except Exception:
                    pass  # skip this one, keep going
            self.done.emit(written)
        except Exception as e:
            self.failed.emit(str(e))


class ComparisonWorker(QObject):
    """Runs the fast multi-portfolio comparison off the UI thread.

    ``named_configs`` is a list of (label, PortfolioConfig). Emits progress per
    portfolio and ``done(list[ComparisonResult])``.
    """

    progress = Signal(str, float)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, named_configs) -> None:
        super().__init__()
        self._named_configs = named_configs

    @Slot()
    def run(self) -> None:
        try:
            from src.analytics import comparison

            results = comparison.compare_portfolios(
                self._named_configs, progress=self._emit
            )
            self.done.emit(results)
        except Exception as e:
            self.failed.emit(str(e))

    def _emit(self, label: str, frac: float) -> None:
        self.progress.emit(str(label), float(frac))


class QuotesWorker(QObject):
    """Fetches a delayed quote snapshot for a set of tickers off the UI thread.

    Drives the always-on ticker strip and the Live Market Watch view. Emits
    ``done(dict[str, Quote])``; never raises (a bad symbol yields an empty
    quote), so ``failed`` only fires on a total fetch collapse.
    """

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, tickers) -> None:
        super().__init__()
        self._tickers = list(tickers or [])

    @Slot()
    def run(self) -> None:
        try:
            from src.data import market_data

            from . import settings

            provider, creds = settings.realtime_provider()
            self.done.emit(
                market_data.fetch_quotes(self._tickers, provider=provider, creds=creds)
            )
        except Exception as e:
            self.failed.emit(str(e))


class WatchlistWorker(QObject):
    """Fetches delayed (or provider) quotes + best-effort names for the Live
    Market Watch watchlist, off the UI thread. Decoupled from the analysis
    pipeline. Emits ``done(dict[str, Quote])``; never raises on a bad symbol."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, tickers, use_cache: bool = True) -> None:
        super().__init__()
        self._tickers = list(tickers or [])
        self._use_cache = bool(use_cache)

    @Slot()
    def run(self) -> None:
        try:
            from src.data import market_data

            from . import settings

            provider, creds = settings.realtime_provider()
            self.done.emit(
                market_data.fetch_quotes(
                    self._tickers,
                    use_cache=self._use_cache,
                    provider=provider,
                    creds=creds,
                    with_names=True,
                )
            )
        except Exception as e:
            self.failed.emit(str(e))


class OhlcWorker(QObject):
    """Fetches one ticker's OHLCV frame at a TradingView-style timeframe off the
    UI thread for the Live Market Watch candlestick chart. Emits
    ``done((ticker, timeframe, DataFrame|None))`` so a stale reply can be matched
    against the currently-selected symbol + timeframe and dropped."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, ticker: str, timeframe: str) -> None:
        super().__init__()
        self._ticker = str(ticker or "")
        self._timeframe = str(timeframe or "")

    @Slot()
    def run(self) -> None:
        try:
            from src.data import market_data

            df = market_data.fetch_ohlc(self._ticker, self._timeframe)
            self.done.emit((self._ticker, self._timeframe, df))
        except Exception as e:
            self.failed.emit(str(e))


class SymbolNewsWorker(QObject):
    """Fetches recent news for a single symbol off the UI thread for the Live
    Market Watch news panel. Emits ``done((ticker, list[NewsItem]))``; never
    raises (a failed fetch yields an empty list)."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, ticker: str) -> None:
        super().__init__()
        self._ticker = str(ticker or "")

    @Slot()
    def run(self) -> None:
        try:
            from src.data import market_data

            from . import settings

            av_key = settings.get_api_key("ALPHAVANTAGE_API_KEY")
            items = market_data.fetch_ticker_news([self._ticker], av_key=av_key,
                                                  per_ticker=12, total_cap=12)
            self.done.emit((self._ticker, items))
        except Exception as e:
            self.failed.emit(str(e))


class MorningReportWorker(QObject):
    """Builds and delivers a daily Morning Report for one portfolio, off the UI
    thread: fetch quotes + calendar + news, render the Morning Brief, optionally
    generate the full analytical report, and optionally email them.

    ``options`` keys: ``config`` (PortfolioConfig), ``name`` (str), ``out_dir``
    (str), ``attach_full`` (bool), ``formats`` (tuple), ``email`` (SmtpConfig or
    None), ``to`` (list[str]). Emits ``done(dict)`` with brief_path / report_paths
    / emailed / email_error / subject / summary; ``failed(str)`` on a hard error
    (a failed *email* is reported inside ``done`` so the report is still saved)."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, options: dict) -> None:
        super().__init__()
        self._o = dict(options)

    @Slot()
    def run(self) -> None:
        from datetime import datetime
        from pathlib import Path

        try:
            from src.data import market_data
            from src.reports.emailer import send_email
            from src.reports.generate import _safe, generate_report
            from src.reports.morning_brief import build_morning_brief

            from . import settings as _settings

            o = self._o
            cfg = o["config"]
            name = o.get("name") or "Portfolio"
            tickers = list(getattr(cfg, "tickers", []) or [])

            provider, creds = _settings.realtime_provider()
            quotes = market_data.fetch_quotes(tickers, provider=provider, creds=creds)
            events = market_data.fetch_calendar(tickers)
            news = market_data.fetch_ticker_news(
                tickers, av_key=_settings.get_api_key("ALPHAVANTAGE_API_KEY")
            )
            brief = build_morning_brief(cfg, quotes, events=events, news=news, name=name)

            out = Path(o["out_dir"])
            out.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d")
            brief_path = out / f"{_safe(name)}_morning_{stamp}.html"
            brief_path.write_text(brief["html"], encoding="utf-8")

            report_paths: list[str] = []
            if o.get("attach_full"):
                report_paths = [
                    str(p) for p in generate_report(
                        cfg, out, formats=o.get("formats", ("pdf",)), name=name
                    )
                ]

            emailed = False
            email_error = None
            email_cfg = o.get("email")
            to = o.get("to") or []
            if email_cfg is not None and to:
                try:
                    send_email(email_cfg, to, brief["subject"], brief["html"],
                               attachments=report_paths or None)
                    emailed = True
                except Exception as e:
                    email_error = str(e)

            self.done.emit({
                "brief_path": str(brief_path),
                "report_paths": report_paths,
                "emailed": emailed,
                "email_error": email_error,
                "subject": brief["subject"],
                "summary": brief["summary"],
            })
        except Exception as e:
            self.failed.emit(str(e))


class StatementsWorker(QObject):
    """Fetches one ticker's financial statements + analyst data off the UI thread.
    Emits ``done((ticker, dict))``."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, ticker) -> None:
        super().__init__()
        self._ticker = ticker

    @Slot()
    def run(self) -> None:
        try:
            from src.data import fundamentals

            self.done.emit((self._ticker, fundamentals.fetch_statements(self._ticker)))
        except Exception as e:
            self.failed.emit(str(e))
