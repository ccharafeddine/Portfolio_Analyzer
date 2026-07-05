"""Theming system for the desktop UI.

A ``Theme`` is a bundle of design tokens (colors, fonts, type sizes, density).
Six built-in themes are provided and can be switched live:

- ``institutional``        — clean, spacious, refined product look (default)
- ``terminal``             — dense, high-contrast, Bloomberg-style power view
- ``minimal``              — understated, generous whitespace, muted palette
- ``light``                — a genuine light/day theme (dark ink on white)
- ``high_contrast_light``  — accessibility variant, black-on-white, AAA
- ``high_contrast_dark``   — accessibility variant, white-on-black, AAA

Widgets read the *active* theme via ``theme.ACTIVE`` so a rebuild picks up the
current tokens. ``stylesheet()`` builds the application-wide QSS from a theme,
and ``chart_palette()`` returns the colors used to theme the Plotly charts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Default categorical chart series — mirrors plotly_charts.COLORS so the
# original themes (Terminal/Institutional/Minimal) keep today's palette when
# they don't declare their own ``chart_series``.
_DEFAULT_SERIES: tuple[str, ...] = (
    "#3B82F6",  # blue
    "#10B981",  # green
    "#F59E0B",  # amber
    "#EF4444",  # red
    "#8B5CF6",  # purple
    "#EC4899",  # pink
    "#06B6D4",  # cyan
    "#F97316",  # orange
)


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

    # Optional / defaulted (kept last so existing themes need no edits).
    chart_series: tuple[str, ...] = _DEFAULT_SERIES  # categorical chart palette
    gloss: bool = False  # glossy accent-button treatment (Frutiger Aero)


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

# ── Aesthetic themes ─────────────────────────────────────────────
# Typography/density mirror INSTITUTIONAL except where noted (radius,
# chart_scale, font, gloss). Each declares its own chart_series.

FRUTIGER = Theme(
    key="frutiger",
    name="Frutiger Aero",
    # Glossy aqua-sky optimism: bright cyan on airy blues, glassy buttons.
    bg="#EAF4FB",
    panel="#F2F8FD",
    card="#FFFFFF",
    card_alt="#E4F1FA",
    border="#CBE1F0",
    border_light="#DCEBF6",
    accent="#0FB0E7",
    accent_hover="#0C97C7",
    accent_text="#0A2230",  # dark ink on the bright glossy aqua (~6.5:1, on-brand)
    text="#10222E",
    text_muted="#5E7A8A",
    text_slate="#3E5A6A",
    green="#1E9E48",  # deepened so gains read on white cards (~3.5:1, was 2.4:1)
    red="#E5484D",
    chart_bg="#FFFFFF",
    chart_grid="rgba(60, 120, 150, 0.12)",
    font="Segoe UI, DM Sans, Helvetica, Arial, sans-serif",
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
    chart_series=(
        "#0FB0E7", "#2FBF57", "#4FC3F7", "#14B8A6",
        "#8BC34A", "#2979FF", "#00BCD4", "#26D0A0",
    ),
    gloss=True,
)

SYNTHWAVE = Theme(
    key="synthwave",
    name="Corporate Synthwave",
    # Deep violet night + magenta laser accent. Mint/rose for up/down so
    # losses never blur into the magenta accent.
    bg="#120726",
    panel="#1B0E38",
    card="#241246",
    card_alt="#2A1650",
    border="#3A1E63",
    border_light="#4E2A80",
    accent="#E23CFF",
    accent_hover="#C21FE0",
    accent_text="#FFFFFF",
    text="#F5E9FF",
    text_muted="#9B7BC7",
    text_slate="#B79BE0",
    green="#2EE6B0",
    red="#FF3B5C",
    chart_bg="#120726",
    chart_grid="rgba(226, 60, 255, 0.14)",  # faint magenta laser-grid tint
    font="DM Sans, Segoe UI, Helvetica, Arial, sans-serif",
    mono="JetBrains Mono, Consolas, monospace",
    base_pt=13,
    label_pt=11,
    heading_pt=15,
    metric_pt=26,
    statval_pt=18,
    radius=8,
    card_pad_v=14,
    card_pad_h=18,
    tab_pad_v=8,
    tab_pad_h=18,
    input_pad=6,
    content_spacing=14,
    chart_scale=1.0,
    chart_series=(
        "#E23CFF", "#22D3EE", "#2EE6B0", "#FF3D9A",
        "#9D5CFF", "#4361FF", "#FFB443", "#FF6B9D",
    ),
    gloss=False,
)

SUPERFLAT = Theme(
    key="superflat",
    name="Superflat Pop",
    # Warm-white gallery walls, candy-pop accents, big flat rounded shapes.
    bg="#FCFCFA",
    panel="#FFFFFF",
    card="#FFFFFF",
    card_alt="#F4F2EC",
    border="#E0DDD3",
    border_light="#ECEAE1",
    accent="#FF3EA5",
    accent_hover="#E62E90",
    accent_text="#FFFFFF",
    text="#16130F",
    text_muted="#6B6459",
    text_slate="#4A443B",
    green="#199444",  # deepened so gains read on white cards (~3.9:1, was 2.4:1)
    red="#FF473E",
    chart_bg="#FFFFFF",
    chart_grid="rgba(20, 20, 20, 0.10)",
    font="DM Sans, Segoe UI, Helvetica, Arial, sans-serif",
    mono="JetBrains Mono, Consolas, monospace",
    base_pt=13,
    label_pt=11,
    heading_pt=15,
    metric_pt=26,
    statval_pt=18,
    radius=14,
    card_pad_v=14,
    card_pad_h=18,
    tab_pad_v=8,
    tab_pad_h=18,
    input_pad=6,
    content_spacing=14,
    chart_scale=1.1,
    chart_series=(
        "#FF3EA5", "#33A1FD", "#FFC61A", "#2FC24E",
        "#FF7A1A", "#9B5DE5", "#FF473E", "#12CDD4",
    ),
    gloss=False,
)

LIGHT = Theme(
    key="light",
    name="Daylight",
    # A genuine light/day theme: near-white surfaces, dark ink, blue accent.
    # bg is the soft page gray; panel/card are raised white so chrome reads as
    # elevated. Borders run slightly darker than the surfaces so edges are
    # visible on white (the inverse of the dark themes, where border_light is
    # the *lighter* of the pair).
    bg="#EEF1F6",
    panel="#FFFFFF",
    card="#FFFFFF",
    card_alt="#F4F6FA",
    border="#E2E8F0",
    border_light="#CBD5E1",
    accent="#2563EB",
    accent_hover="#1D4ED8",
    accent_text="#FFFFFF",
    text="#0F172A",       # ~16:1 on white
    text_muted="#4D596D",  # ~7.1:1 on white (AAA for text)
    text_slate="#475569",  # ~7.5:1 on white
    green="#059669",
    red="#DC2626",
    chart_bg="#FFFFFF",
    chart_grid="rgba(15, 23, 42, 0.10)",
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

HIGH_CONTRAST_LIGHT = Theme(
    key="high_contrast_light",
    name="High Contrast (Light)",
    # Accessibility variant: pure black-on-white, saturated accent, heavy
    # borders. Every foreground pair clears WCAG AAA (7:1+) against white.
    bg="#FFFFFF",
    panel="#FFFFFF",
    card="#FFFFFF",
    card_alt="#F2F2F2",
    border="#000000",
    border_light="#333333",
    accent="#0B5FD4",      # ~5.9:1 on white; white text ~5.9:1 on accent
    accent_hover="#0847A6",
    accent_text="#FFFFFF",
    text="#000000",
    text_muted="#3A3A3A",  # ~10:1 on white
    text_slate="#1A1A1A",
    green="#0F7A34",       # ~5:1 on white
    red="#C21818",         # ~6:1 on white
    chart_bg="#FFFFFF",
    chart_grid="rgba(0, 0, 0, 0.22)",
    font="DM Sans, Segoe UI, Helvetica, Arial, sans-serif",
    mono="JetBrains Mono, Consolas, monospace",
    base_pt=14,
    label_pt=12,
    heading_pt=16,
    metric_pt=27,
    statval_pt=19,
    radius=6,
    card_pad_v=14,
    card_pad_h=18,
    tab_pad_v=8,
    tab_pad_h=18,
    input_pad=6,
    content_spacing=14,
    chart_scale=1.0,
)

HIGH_CONTRAST_DARK = Theme(
    key="high_contrast_dark",
    name="High Contrast (Dark)",
    # Accessibility variant: pure white-on-black, amber accent, heavy white
    # borders. Every foreground pair clears WCAG AAA (7:1+) against black.
    bg="#000000",
    panel="#000000",
    card="#000000",
    card_alt="#1A1A1A",
    border="#FFFFFF",
    border_light="#CCCCCC",
    accent="#FFD400",      # ~14.7:1 on black; black text ~14.7:1 on accent
    accent_hover="#E6BE00",
    accent_text="#000000",
    text="#FFFFFF",
    text_muted="#A6A6A6",  # ~8.6:1 on black
    text_slate="#CCCCCC",  # ~13:1 on black
    green="#4AE88A",       # ~13:1 on black
    red="#FF6B6B",         # ~7.6:1 on black
    chart_bg="#000000",
    chart_grid="rgba(255, 255, 255, 0.22)",
    font="DM Sans, Segoe UI, Helvetica, Arial, sans-serif",
    mono="JetBrains Mono, Consolas, monospace",
    base_pt=14,
    label_pt=12,
    heading_pt=16,
    metric_pt=27,
    statval_pt=19,
    radius=6,
    card_pad_v=14,
    card_pad_h=18,
    tab_pad_v=8,
    tab_pad_h=18,
    input_pad=6,
    content_spacing=14,
    chart_scale=1.0,
)

THEMES: dict[str, Theme] = {
    t.key: t
    for t in (
        TERMINAL, INSTITUTIONAL, MINIMAL,
        FRUTIGER, SYNTHWAVE, SUPERFLAT,
        LIGHT, HIGH_CONTRAST_LIGHT, HIGH_CONTRAST_DARK,
    )
}

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


def _luminance(hex_color: str) -> float:
    """Relative luminance (0=black, 1=white) of a #RRGGBB color."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return 0.0
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _lighten(hex_color: str, amount: float) -> str:
    """Blend a #RRGGBB color toward white by ``amount`` (0..1)."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


def is_light_theme(theme: Theme | None = None) -> bool:
    """True when the theme's chart background is light enough to want a light
    Plotly template (``plotly_white``) rather than the dark default."""
    t = theme or ACTIVE
    return _luminance(t.chart_bg) > 0.5


def chart_palette(theme: Theme | None = None) -> dict:
    """Colors passed to plotly_charts.apply_palette() so charts match the theme.

    Includes the theme's categorical ``series`` and an ``is_light`` flag
    (derived from chart-background luminance, not hardcoded per key) so the
    chart layer can flip to the light Plotly template.
    """
    t = theme or ACTIVE
    return dict(
        bg=t.chart_bg,
        paper=t.chart_bg,
        grid=t.chart_grid,
        text=t.text,
        muted=t.text_slate,
        card=t.card,
        border=t.border_light,
        series=t.chart_series,
        is_light=is_light_theme(t),
    )


def stylesheet(theme: Theme | None = None) -> str:
    t = theme or ACTIVE

    # Glossy accent-button treatment (Frutiger Aero only). A lighter tint of
    # the accent at the top fading to the accent at the bottom, plus a faint
    # top highlight — applied to the primary button and accent menu selection.
    # Every other theme keeps a flat accent fill.
    if t.gloss:
        _btn = (
            f"qlineargradient(x1:0, y1:0, x2:0, y2:1, "
            f"stop:0 {_lighten(t.accent, 0.38)}, stop:1 {t.accent})"
        )
        _btn_hover = (
            f"qlineargradient(x1:0, y1:0, x2:0, y2:1, "
            f"stop:0 {_lighten(t.accent_hover, 0.38)}, stop:1 {t.accent_hover})"
        )
        _btn_top = f"border-top: 1px solid {_lighten(t.accent, 0.6)};"
    else:
        _btn = t.accent
        _btn_hover = t.accent_hover
        _btn_top = ""

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
    QMenu::item:selected {{ background: {_btn}; color: {t.accent_text}; }}
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
    QWidget#sidebarBrand {{ border-bottom: 1px solid {t.border}; }}
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
        background: {_btn};
        color: {t.accent_text};
        border: none;
        {_btn_top}
        border-radius: {t.radius}px;
        padding: {t.input_pad + 3}px {t.input_pad + 12}px;
        font-weight: 600;
    }}
    QPushButton:hover {{ background: {_btn_hover}; }}
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
