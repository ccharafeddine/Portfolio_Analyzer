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

            self.done.emit(market_data.fetch_quotes(self._tickers))
        except Exception as e:
            self.failed.emit(str(e))


class IntradayWorker(QObject):
    """Fetches one ticker's 1-minute intraday frame off the UI thread for the
    Live Market Watch click-through chart. Emits ``done((ticker, DataFrame|None))``."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, ticker: str) -> None:
        super().__init__()
        self._ticker = str(ticker or "")

    @Slot()
    def run(self) -> None:
        try:
            from src.data import market_data

            self.done.emit((self._ticker, market_data.fetch_intraday(self._ticker)))
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
