"""Theme registry + chart-palette invariants.

These guard the theming contract the desktop UI and Plotly chart layer depend
on: every registered theme carries a usable categorical series, light themes
are detected by luminance (so charts flip to the light Plotly template), and
building a theme's QSS / chart palette never raises.
"""

from src.ui import theme


def test_every_theme_has_enough_chart_series():
    assert theme.THEMES, "no themes registered"
    for key, t in theme.THEMES.items():
        assert len(t.chart_series) >= 6, f"{key} has too few chart series colors"


def test_light_theme_resolves_is_light():
    # Frutiger Aero and Superflat Pop are genuine light themes (white charts).
    assert theme.is_light_theme(theme.THEMES["frutiger"]) is True
    assert theme.is_light_theme(theme.THEMES["superflat"]) is True
    # A dark theme must not be mistaken for light.
    assert theme.is_light_theme(theme.THEMES["synthwave"]) is False
    assert theme.is_light_theme(theme.THEMES["terminal"]) is False


def test_new_aesthetic_themes_registered():
    for key in ("frutiger", "synthwave", "superflat"):
        assert key in theme.THEMES


def test_chart_palette_and_stylesheet_build_for_every_theme():
    for key, t in theme.THEMES.items():
        pal = theme.chart_palette(t)
        # chart_palette exposes the series + light flag the chart layer needs.
        assert pal["series"] == t.chart_series
        assert isinstance(pal["is_light"], bool)
        assert set(pal) >= {"bg", "paper", "grid", "text", "muted", "series", "is_light"}
        qss = theme.stylesheet(t)
        assert isinstance(qss, str) and qss.strip(), f"{key} produced empty QSS"


def test_only_gloss_themes_emit_a_gradient_button():
    # Frutiger is the sole glossy theme; its QSS carries a gradient, others flat.
    assert "qlineargradient" in theme.stylesheet(theme.THEMES["frutiger"])
    assert "qlineargradient" not in theme.stylesheet(theme.THEMES["terminal"])
