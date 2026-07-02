"""Desktop UI package (PySide6) for Portfolio Analyzer.

This package is purely additive: it consumes the UI-agnostic compute layer
under ``src/`` (pipeline, analytics, charts, reports, config) and renders it
in a native desktop application. Nothing in ``src.analytics``/``src.pipeline``
imports from here.
"""

__app_name__ = "Portfolio Analyzer"
__version__ = "2.0.0"
