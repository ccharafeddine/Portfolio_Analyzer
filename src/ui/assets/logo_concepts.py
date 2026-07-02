"""Logo concept SVGs + a renderer to preview them as PNGs.

Run with the project venv to (re)generate preview PNGs into a target dir:
    python -m src.ui.assets.logo_concepts <out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

# Shared defs: dark rounded tile + accent gradient.
_DEFS = """
  <defs>
    <linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0' stop-color='#101826'/>
      <stop offset='1' stop-color='#070B12'/>
    </linearGradient>
    <linearGradient id='acc' x1='0' y1='1' x2='1' y2='0'>
      <stop offset='0' stop-color='#2563EB'/>
      <stop offset='1' stop-color='#22D3EE'/>
    </linearGradient>
  </defs>
"""

_TILE = "<rect x='8' y='8' width='240' height='240' rx='54' fill='url(#bg)' stroke='#1E293B' stroke-width='2'/>"


# Concept A — ascending bars with a trend line and end node.
CONCEPT_A = f"""<svg width='256' height='256' viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'>
{_DEFS}
  {_TILE}
  <rect x='66' y='150' width='30' height='54' rx='9' fill='url(#acc)' opacity='0.5'/>
  <rect x='113' y='120' width='30' height='84' rx='9' fill='url(#acc)' opacity='0.72'/>
  <rect x='160' y='90' width='30' height='114' rx='9' fill='url(#acc)'/>
  <polyline points='64,150 112,118 158,132 196,80' fill='none' stroke='#E8EEF6'
    stroke-width='6' stroke-linecap='round' stroke-linejoin='round'/>
  <circle cx='196' cy='80' r='11' fill='#22D3EE' stroke='#0B1120' stroke-width='4'/>
</svg>"""


# Concept B — allocation donut (portfolio weights) with a highlighted segment.
CONCEPT_B = f"""<svg width='256' height='256' viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'>
{_DEFS}
  {_TILE}
  <g transform='rotate(-90 128 128)'>
    <circle cx='128' cy='128' r='62' fill='none' stroke='#22D3EE' stroke-width='30'
      opacity='0.22'/>
    <circle cx='128' cy='128' r='62' fill='none' stroke='url(#acc)' stroke-width='30'
      stroke-dasharray='156 234' stroke-linecap='butt'/>
    <circle cx='128' cy='128' r='62' fill='none' stroke='#E8EEF6' stroke-width='30'
      stroke-dasharray='58 331' stroke-dashoffset='-160' stroke-linecap='butt' opacity='0.85'/>
  </g>
  <circle cx='128' cy='128' r='30' fill='#0B1120'/>
  <polyline points='114,134 124,124 140,140 156,116' fill='none' stroke='#22D3EE'
    stroke-width='6' stroke-linecap='round' stroke-linejoin='round' transform='translate(-6 -2)'/>
</svg>"""


# Concept C — minimal line/area 'peak' with a data node.
CONCEPT_C = f"""<svg width='256' height='256' viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'>
{_DEFS}
  {_TILE}
  <path d='M56 176 L104 140 L140 158 L200 88 L200 200 L56 200 Z' fill='url(#acc)' opacity='0.18'/>
  <polyline points='56,176 104,140 140,158 200,88' fill='none' stroke='url(#acc)'
    stroke-width='8' stroke-linecap='round' stroke-linejoin='round'/>
  <circle cx='104' cy='140' r='9' fill='#0B1120' stroke='#22D3EE' stroke-width='5'/>
  <circle cx='200' cy='88' r='11' fill='#22D3EE' stroke='#0B1120' stroke-width='4'/>
</svg>"""


CONCEPTS = {"A_bars": CONCEPT_A, "B_monogram": CONCEPT_B, "C_peak": CONCEPT_C}


def render_pngs(out_dir: Path, size: int = 256) -> list[Path]:
    from PySide6.QtCore import QByteArray, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, svg in CONCEPTS.items():
        (out_dir / f"logo_{name}.svg").write_text(svg, encoding="utf-8")
        renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
        img = QImage(size, size, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        painter = QPainter(img)
        renderer.render(painter)
        painter.end()
        path = out_dir / f"logo_{name}.png"
        img.save(str(path), "PNG")
        written.append(path)
    return written


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    for p in render_pngs(out):
        print(p)
