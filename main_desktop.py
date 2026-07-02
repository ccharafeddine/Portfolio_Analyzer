"""Portfolio Analyzer — native desktop entry point.

Launches the PySide6 application. Qt/QtWebEngine attributes that must be set
*before* the QApplication is constructed are handled here.

Run locally with:  python main_desktop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable when run as a script or when frozen.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    # Point the price cache at the app's cross-platform cache dir before any
    # module that reads PORTFOLIO_ANALYZER_CACHE_DIR is imported.
    import os

    from src.ui import paths

    os.environ.setdefault("PORTFOLIO_ANALYZER_CACHE_DIR", str(paths.cache_dir()))

    # On Windows, set an explicit AppUserModelID so the taskbar shows the app's
    # own icon (the logo) instead of the generic python.exe icon.
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "PortfolioAnalyzer.Desktop"
            )
        except Exception:
            pass

    # QtWebEngine requires a shared OpenGL context set before QApplication.
    from PySide6.QtCore import Qt, QCoreApplication

    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    from PySide6.QtWidgets import QApplication

    from PySide6.QtGui import QIcon

    from src.ui import __app_name__, __version__
    from src.ui.assets import mark_path
    from src.ui.settings import ORG_NAME, APP_NAME
    from src.ui.main_window import MainWindow

    QCoreApplication.setOrganizationName(ORG_NAME)
    QCoreApplication.setApplicationName(APP_NAME)
    QCoreApplication.setApplicationVersion(__version__)

    app = QApplication(sys.argv)
    app.setApplicationDisplayName(__app_name__)
    app.setWindowIcon(QIcon(mark_path()))

    window = MainWindow()
    window.show()
    window.maybe_show_onboarding()
    return app.exec()


if __name__ == "__main__":
    # Headless report mode (for schedulers): PortfolioAnalyzer.exe --generate-report ...
    if "--generate-report" in sys.argv:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.report_cli import main as report_main

        args = [a for a in sys.argv[1:] if a != "--generate-report"]
        raise SystemExit(report_main(args))
    raise SystemExit(main())
