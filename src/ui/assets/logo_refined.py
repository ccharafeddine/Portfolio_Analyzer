"""Refined donut logo variants (bolder, more iconic, legible when small).

    python -m src.ui.assets.logo_refined <out_dir>
renders each at 256px and 44px (to check small-icon legibility).
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEFS = """
  <defs>
    <linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0' stop-color='#111C2E'/>
      <stop offset='1' stop-color='#070B12'/>
    </linearGradient>
    <linearGradient id='acc' x1='0' y1='1' x2='1' y2='0'>
      <stop offset='0' stop-color='#2563EB'/>
      <stop offset='1' stop-color='#38D6F0'/>
    </linearGradient>
  </defs>
"""
_TILE = "<rect x='6' y='6' width='244' height='244' rx='56' fill='url(#bg)' stroke='#22304A' stroke-width='2'/>"

# V1 — bold allocation ring + upward arrow (allocation + growth).
V1 = f"""<svg width='256' height='256' viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'>
{_DEFS}{_TILE}
  <g transform='rotate(-90 128 128)'>
    <circle cx='128' cy='128' r='66' fill='none' stroke='#2A3A57' stroke-width='36'/>
    <circle cx='128' cy='128' r='66' fill='none' stroke='url(#acc)' stroke-width='36'
      stroke-dasharray='187 228'/>
    <circle cx='128' cy='128' r='66' fill='none' stroke='#E9F1FA' stroke-width='36'
      stroke-dasharray='75 340' stroke-dashoffset='-190' opacity='0.9'/>
  </g>
  <path d='M128 96 L156 132 L138 132 L138 162 L118 162 L118 132 L100 132 Z' fill='#E9F1FA'/>
</svg>"""

# V2 — clean single-wedge ring, hollow center (minimal, most legible).
V2 = f"""<svg width='256' height='256' viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'>
{_DEFS}{_TILE}
  <g transform='rotate(-90 128 128)'>
    <circle cx='128' cy='128' r='66' fill='none' stroke='#2A3A57' stroke-width='34'/>
    <circle cx='128' cy='128' r='66' fill='none' stroke='url(#acc)' stroke-width='34'
      stroke-dasharray='145 270' stroke-linecap='round'/>
  </g>
  <circle cx='128' cy='62' r='9' fill='#38D6F0'/>
</svg>"""

# V3 — allocation ring + bold ascending bars in the center.
V3 = f"""<svg width='256' height='256' viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'>
{_DEFS}{_TILE}
  <g transform='rotate(-90 128 128)'>
    <circle cx='128' cy='128' r='66' fill='none' stroke='#2A3A57' stroke-width='30'/>
    <circle cx='128' cy='128' r='66' fill='none' stroke='url(#acc)' stroke-width='30'
      stroke-dasharray='200 215'/>
  </g>
  <g>
    <rect x='98'  y='140' width='16' height='22' rx='4' fill='#E9F1FA' opacity='0.6'/>
    <rect x='120' y='126' width='16' height='36' rx='4' fill='#E9F1FA' opacity='0.8'/>
    <rect x='142' y='110' width='16' height='52' rx='4' fill='#38D6F0'/>
  </g>
</svg>"""

VARIANTS = {"V1_arrow": V1, "V2_wedge": V2, "V3_bars": V3}


def render(out_dir: Path) -> list[Path]:
    from PySide6.QtCore import QByteArray, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, svg in VARIANTS.items():
        (out_dir / f"refined_{name}.svg").write_text(svg, encoding="utf-8")
        renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
        for size in (256, 44):
            img = QImage(size, size, QImage.Format_ARGB32)
            img.fill(Qt.transparent)
            p = QPainter(img)
            renderer.render(p)
            p.end()
            path = out_dir / f"refined_{name}_{size}.png"
            img.save(str(path), "PNG")
            written.append(path)
    return written


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    for p in render(out):
        print(p)
