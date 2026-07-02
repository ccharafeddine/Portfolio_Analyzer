"""Theming system for the desktop UI.

A ``Theme`` is a bundle of design tokens (colors, fonts, type sizes, density).
Three built-in themes are provided and can be switched live:

- ``institutional`` — clean, spacious, refined product look (default)
- ``terminal``      — dense, high-contrast, Bloomberg-style power view
- ``minimal``       — understated, generous whitespace, muted palette

Widgets read the *active* theme via ``theme.ACTIVE`` so a rebuild picks up the
current tokens. ``stylesheet()`` builds the application-wide QSS from a theme,
and ``chart_palette()`` returns the colors used to theme the Plotly charts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Theme:
    key: str
    name: str

    # Colors
    bg: str
    panel: str
    card: str
    card_alt: str
    border: str
    border_light: str
    accent: str
    accent_hover: str
    accent_text: str  # text color on top of accent (buttons)
    text: str
    text_muted: str
    text_slate: str
    green: str
    red: str
    # Chart-specific
    chart_bg: str
    chart_grid: str

    # Typography
    font: str
    mono: str
    base_pt: int
    label_pt: int
    heading_pt: int
    metric_pt: int
    statval_pt: int

    # Density
    radius: int
    card_pad_v: int
    card_pad_h: int
    tab_pad_v: int
    tab_pad_h: int
    input_pad: int
    content_spacing: int
    chart_scale: float  # multiplies base chart heights


INSTITUTIONAL = Theme(
    key="institutional",
    name="Modern Institutional",
    bg="#0B1120",
    panel="#0D1526",
    card="#151D2E",
    card_alt="#1A2438",
    border="#1E293B",
    border_light="#2D3A50",
    accent="#3B82F6",
    accent_hover="#2563EB",
    accent_text="#FFFFFF",
    text="#F1F5F9",
    text_muted="#64748B",
    text_slate="#94A3B8",
    green="#10B981",
    red="#EF4444",
    chart_bg="#0B1120",
    chart_grid="rgba(148, 163, 184, 0.08)",
    font="DM Sans, Segoe UI, Helvetica, Arial, sans-serif",
    mono="JetBrains Mono, Consolas, monospace",
    base_pt=13,
    label_pt=11,
    heading_pt=15,
    metric_pt=26,
    statval_pt=18,
    radius=12,
    card_pad_v=14,
    card_pad_h=18,
    tab_pad_v=8,
    tab_pad_h=18,
    input_pad=6,
    content_spacing=14,
    chart_scale=1.0,
)

TERMINAL = Theme(
    key="terminal",
    name="Bloomberg Terminal",
    bg="#05070D",
    panel="#0A0E16",
    card="#0E141F",
    card_alt="#121A28",
    border="#1B2433",
    border_light="#28374B",
    accent="#F59E0B",
    accent_hover="#D97706",
    accent_text="#0A0E16",
    text="#E6EDF3",
    text_muted="#6B7688",
    text_slate="#8B97A8",
    green="#26D07C",
    red="#FF5C5C",
    chart_bg="#05070D",
    chart_grid="rgba(139, 151, 168, 0.10)",
    font="Segoe UI, Helvetica, Arial, sans-serif",
    mono="JetBrains Mono, Consolas, monospace",
    base_pt=11,
    label_pt=10,
    heading_pt=12,
    metric_pt=20,
    statval_pt=15,
    radius=3,
    card_pad_v=9,
    card_pad_h=12,
    tab_pad_v=5,
    tab_pad_h=12,
    input_pad=4,
    content_spacing=9,
    chart_scale=0.85,
)

MINIMAL = Theme(
    key="minimal",
    name="Minimal Premium",
    # Neutral warm charcoal (no blue tint) + teal accent — deliberately distinct
    # from the navy/blue Institutional theme.
    bg="#15171C",
    panel="#1A1D23",
    card="#1F232A",
    card_alt="#242932",
    border="#2A2F38",
    border_light="#333A45",
    accent="#4FD1C5",
    accent_hover="#38B2AC",
    accent_text="#0E1417",
    text="#EAECEF",
    text_muted="#808894",
    text_slate="#9BA3AE",
    green="#4ADE80",
    red="#F87171",
    chart_bg="#15171C",
    chart_grid="rgba(155, 163, 174, 0.06)",
    font="DM Sans, Segoe UI, Helvetica, Arial, sans-serif",
    mono="JetBrains Mono, Consolas, monospace",
    base_pt=14,
    label_pt=11,
    heading_pt=20,
    metric_pt=30,
    statval_pt=20,
    radius=14,
    card_pad_v=20,
    card_pad_h=24,
    tab_pad_v=10,
    tab_pad_h=22,
    input_pad=8,
    content_spacing=20,
    chart_scale=1.15,
)

THEMES: dict[str, Theme] = {t.key: t for t in (TERMINAL, INSTITUTIONAL, MINIMAL)}

# ── UI scale (zoom) ──
# A global multiplier applied to every font size / padding on top of the base
# theme, so users can enlarge the whole app. 1.1 is the default (slightly bigger
# than the base design). Presets are surfaced in View → Text Size.
DEFAULT_SCALE = 1.25
MIN_SCALE = 0.8
MAX_SCALE = 1.8
SCALE_PRESETS = [
    ("Small", 0.9),
    ("Normal", 1.0),
    ("Large", 1.1),
    ("Larger", 1.25),
    ("Largest", 1.5),
]

_BASE: Theme = TERMINAL  # the selected base theme, before scaling
SCALE: float = DEFAULT_SCALE

# Size/spacing fields multiplied by SCALE (colors/fonts untouched).
_SCALED_FIELDS = (
    "base_pt", "label_pt", "heading_pt", "metric_pt", "statval_pt",
    "card_pad_v", "card_pad_h", "tab_pad_v", "tab_pad_h", "input_pad",
    "content_spacing", "radius",
)


def _scaled(theme: Theme, factor: float) -> Theme:
    if factor == 1.0:
        return theme
    changes = {f: max(1, round(getattr(theme, f) * factor)) for f in _SCALED_FIELDS}
    changes["chart_scale"] = theme.chart_scale * factor
    return replace(theme, **changes)


# The active theme = selected base theme, scaled. Widgets read this at build time.
ACTIVE: Theme = _scaled(_BASE, SCALE)


def set_active(key: str) -> Theme:
    global _BASE, ACTIVE
    _BASE = THEMES.get(key, TERMINAL)
    ACTIVE = _scaled(_BASE, SCALE)
    return ACTIVE


def set_scale(factor: float) -> float:
    global SCALE, ACTIVE
    SCALE = max(MIN_SCALE, min(MAX_SCALE, float(factor)))
    ACTIVE = _scaled(_BASE, SCALE)
    return SCALE


def current_scale() -> float:
    return SCALE


def base_key() -> str:
    return _BASE.key


def chart_palette(theme: Theme | None = None) -> dict:
    """Colors passed to plotly_charts.apply_palette() so charts match the theme."""
    t = theme or ACTIVE
    return dict(
        bg=t.chart_bg,
        paper=t.chart_bg,
        grid=t.chart_grid,
        text=t.text,
        muted=t.text_slate,
    )


def stylesheet(theme: Theme | None = None) -> str:
    t = theme or ACTIVE
    return f"""
    QMainWindow, QWidget {{
        background-color: {t.bg};
        color: {t.text};
        font-family: {t.font};
        font-size: {t.base_pt}px;
    }}

    QMenuBar {{ background-color: {t.panel}; border-bottom: 1px solid {t.border}; padding: 2px; }}
    QMenuBar::item {{ padding: 6px 12px; background: transparent; }}
    QMenuBar::item:selected {{ background-color: {t.card}; border-radius: 4px; }}
    QMenu {{ background-color: {t.panel}; border: 1px solid {t.border}; }}
    QMenu::item {{ padding: 6px 24px; }}
    QMenu::item:selected {{ background-color: {t.accent}; color: {t.accent_text}; }}
    QMenu::indicator:checked {{ color: {t.accent}; }}

    QDockWidget {{ color: {t.text_slate}; }}
    QDockWidget::title {{
        background-color: {t.panel};
        padding: 8px 12px;
        border-bottom: 1px solid {t.border};
        text-transform: uppercase;
        font-size: {t.label_pt}px;
        letter-spacing: 0.06em;
    }}

    QToolButton#collapsibleHeader {{
        background: transparent; border: none; color: {t.text_muted};
        font-size: {t.label_pt}px; font-weight: 700; letter-spacing: 0.07em;
        padding: 4px 0; margin-top: 4px;
    }}
    QToolButton#collapsibleHeader:hover {{ color: {t.text}; }}

    QFrame#sidebar {{ background-color: {t.panel}; border-right: 1px solid {t.border}; }}
    QWidget#sidebarHeader {{ border-bottom: 1px solid {t.border}; }}
    QLabel#sidebarTitle {{
        color: {t.text_muted}; font-size: {t.label_pt}px; font-weight: 700;
        letter-spacing: 0.07em;
    }}
    QToolButton#sidebarToggle {{
        background: {t.card}; color: {t.text_slate}; border: 1px solid {t.border_light};
        border-radius: {max(4, t.radius - 4)}px; padding: 2px 8px; font-size: {t.heading_pt}px;
        font-weight: 700;
    }}
    QToolButton#sidebarToggle:hover {{ border-color: {t.accent}; color: {t.text}; }}

    QTabWidget::pane {{ border: none; border-top: 1px solid {t.border}; }}
    QTabBar::tab {{
        background: transparent;
        color: {t.text_muted};
        padding: {t.tab_pad_v}px {t.tab_pad_h}px;
        margin-right: 2px;
        border: none;
        font-weight: 500;
    }}
    QTabBar::tab:selected {{ color: {t.text}; border-bottom: 2px solid {t.accent}; }}
    QTabBar::tab:hover {{ color: {t.text_slate}; }}

    QPushButton {{
        background-color: {t.accent};
        color: {t.accent_text};
        border: none;
        border-radius: {t.radius}px;
        padding: {t.input_pad + 3}px {t.input_pad + 12}px;
        font-weight: 600;
    }}
    QPushButton:hover {{ background-color: {t.accent_hover}; }}
    QPushButton:disabled {{ background-color: {t.border_light}; color: {t.text_muted}; }}
    QPushButton#secondary {{
        background-color: {t.card};
        border: 1px solid {t.border_light};
        color: {t.text_slate};
    }}
    QPushButton#secondary:hover {{ border-color: {t.accent}; color: {t.text}; }}

    QLineEdit, QPlainTextEdit, QTextEdit, QDateEdit, QDoubleSpinBox, QSpinBox {{
        background-color: {t.card};
        border: 1px solid {t.border_light};
        border-radius: {max(4, t.radius - 4)}px;
        padding: {t.input_pad}px {t.input_pad + 2}px;
        color: {t.text};
        selection-background-color: {t.accent};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QDateEdit:focus,
    QDoubleSpinBox:focus, QSpinBox:focus {{ border-color: {t.accent}; }}

    QGroupBox {{
        border: 1px solid {t.border};
        border-radius: {max(4, t.radius - 2)}px;
        margin-top: 10px;
        padding: 10px;
        color: {t.text_slate};
    }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}

    QToolTip {{
        background-color: {t.panel};
        color: {t.text};
        border: 1px solid {t.accent};
        border-radius: 6px;
        padding: 8px 10px;
        font-size: {t.base_pt}px;
    }}

    QCheckBox {{ spacing: 8px; color: {t.text_slate}; }}
    QLabel {{ color: {t.text}; background: transparent; }}
    QLabel#muted {{ color: {t.text_muted}; font-size: {t.base_pt - 1}px; }}
    QLabel#sectionLabel {{
        color: {t.text_muted};
        font-size: {t.label_pt}px;
        font-weight: 700;
        letter-spacing: 0.07em;
    }}

    QTableView {{
        background-color: {t.card};
        alternate-background-color: {t.card_alt};
        gridline-color: {t.border};
        border: 1px solid {t.border};
        border-radius: {max(4, t.radius - 4)}px;
        selection-background-color: {t.accent};
    }}
    QHeaderView::section {{
        background-color: {t.panel};
        color: {t.text_muted};
        padding: {t.input_pad}px {t.input_pad + 4}px;
        border: none;
        border-right: 1px solid {t.border};
        border-bottom: 1px solid {t.border};
        font-weight: 600;
        text-transform: uppercase;
        font-size: {t.label_pt}px;
    }}

    QScrollArea {{ border: none; background: {t.bg}; }}
    QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {t.border_light}; border-radius: 4px; min-height: 28px; }}
    QScrollBar::handle:vertical:hover {{ background: {t.accent}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: {t.border_light}; border-radius: 4px; min-width: 28px; }}
    QScrollBar::handle:horizontal:hover {{ background: {t.accent}; }}

    QStatusBar {{ background-color: {t.panel}; border-top: 1px solid {t.border}; color: {t.text_slate}; }}
    QStatusBar::item {{ border: none; }}
    QLabel#statusLabel {{ color: {t.text_slate}; padding-left: 6px; }}
    QProgressBar {{
        background-color: {t.card};
        border: none;
        border-radius: 0px;
        text-align: center;
        color: {t.text};
        font-weight: 600;
        font-size: {t.label_pt}px;
    }}
    QProgressBar::chunk {{ background-color: {t.accent}; }}
    """
