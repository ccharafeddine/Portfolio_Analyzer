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
    """Fetches per-holding news off the UI thread. Emits ``done(list[NewsItem])``."""

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

            self.done.emit(market_data.fetch_ticker_news(self._tickers, av_key=self._av_key))
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
