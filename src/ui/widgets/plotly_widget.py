"""A QWidget that renders a Plotly figure in an embedded web view.

Design notes
------------
- **Offline**: plotly.js is written once to a local resources folder and loaded
  from there (no CDN), so charts work with no network.
- **Seamless updates**: the view loads a stable shell (plotly.js + an empty
  graph div) exactly once. Setting a figure calls ``Plotly.react`` via
  JavaScript — the page never reloads, so there is no flash when results change
  or the view is reused across runs.
- **No white flash**: the container and web page backgrounds are the theme's
  chart background.
- Charts are reused verbatim from ``src/charts/plotly_charts.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QUrl
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .. import paths
from .. import theme

_PLOTLY_JS_NAME = "plotly.min.js"


def _web_resources_dir() -> Path:
    d = paths.data_dir() / "web"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_plotly_js() -> Path:
    """Write the offline plotly.js bundle into the resources dir once."""
    target = _web_resources_dir() / _PLOTLY_JS_NAME
    if not target.exists() or target.stat().st_size == 0:
        from plotly.offline import get_plotlyjs

        target.write_text(get_plotlyjs(), encoding="utf-8")
    return target


class PlotlyWidget(QWidget):
    """Displays a single Plotly figure. Call :meth:`set_figure` to (re)render."""

    _CONFIG_JS = "{responsive:true, displaylogo:false, displayModeBar:true}"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        _ensure_plotly_js()
        self._base_url = QUrl.fromLocalFile(str(_web_resources_dir()) + "/")

        bg = QColor(theme.ACTIVE.chart_bg)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.Window, bg)
        self.setPalette(pal)

        self._view = QWebEngineView(self)
        self._view.page().setBackgroundColor(bg)
        self._view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.ShowScrollBars, False
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._loaded = False
        self._pending_json: Optional[str] = None
        self._view.loadFinished.connect(self._on_loaded)
        self._view.setHtml(self._shell_html(), self._base_url)

    def _shell_html(self) -> str:
        bg = theme.ACTIVE.chart_bg
        return f"""<!DOCTYPE html><html><head>
<meta charset='utf-8'/>
<script src='{_PLOTLY_JS_NAME}'></script>
<style>
  html,body{{height:100%;margin:0;overflow:hidden;background:{bg}}}
  #gd{{width:100%;height:100%}}
</style></head>
<body><div id='gd'></div>
<script>
  function renderFig(fig){{
    Plotly.react('gd', fig.data, fig.layout, {self._CONFIG_JS});
  }}
</script></body></html>"""

    def _on_loaded(self, ok: bool) -> None:
        self._loaded = True
        if self._pending_json is not None:
            self._push(self._pending_json)
            self._pending_json = None

    def _push(self, fig_json: str) -> None:
        self._view.page().runJavaScript(f"renderFig({fig_json})")

    def set_figure(self, fig) -> None:
        """Render a plotly ``go.Figure`` via Plotly.react (no page reload)."""
        if fig is None:
            if self._loaded:
                self._view.page().runJavaScript("Plotly.purge('gd')")
            self._pending_json = None
            return
        # Fill the container; drop any fixed pixel height from the chart function.
        fig.update_layout(autosize=True, height=None)
        fig_json = fig.to_json()
        if self._loaded:
            self._push(fig_json)
        else:
            self._pending_json = fig_json
