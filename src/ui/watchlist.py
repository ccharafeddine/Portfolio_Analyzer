"""Persistent watchlist for the Live Market Watch section.

An ordered, de-duplicated list of normalized symbols, persisted under
``watchlist/symbols`` in the app's ``QSettings`` store (the same store
``AppSettings`` wraps). Fully decoupled from the analysis pipeline.

Qt-free at import: ``QSettings`` is only imported lazily inside the default
backend, so the store logic is unit-testable on CI's headless runner. Tests
inject a plain dict-backed settings object exposing ``value``/``setValue``.
"""

from __future__ import annotations

from typing import Optional

from src.data.market_data import normalize_symbol

# Persisted keys. INIT_KEY is a scalar sentinel used purely to detect first run,
# which sidesteps QSettings' unreliable empty-list round-tripping (an empty saved
# watchlist and a never-saved one would otherwise look identical).
SYMBOLS_KEY = "watchlist/symbols"
INIT_KEY = "watchlist/initialized"
INDICES_KEY = "watchlist/indices_seeded"

# Market-context indices seeded at the front of every watchlist (once).
INDEX_STARTERS = ["^GSPC", "^IXIC", "^DJI", "^VIX"]
# First-run starters (only ever seeded once), mirroring the sample-portfolio seed.
STARTERS = INDEX_STARTERS + ["SPY", "AAPL", "BTC-USD"]


def _default_settings():
    from PySide6.QtCore import QSettings

    from .settings import APP_NAME, ORG_NAME

    return QSettings(ORG_NAME, APP_NAME)


def _coerce_list(raw) -> list[str]:
    """Normalize whatever QSettings hands back into a list of strings.

    QSettings may return ``None`` (missing), a bare ``str`` (single-element
    list on some platforms), or a real list/tuple.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        return [raw]
    try:
        return [str(x) for x in raw]
    except TypeError:
        return [str(raw)]


def _dedupe(symbols) -> list[str]:
    """Normalize, drop blanks, and de-duplicate while preserving order."""
    out: list[str] = []
    for s in symbols:
        sym = normalize_symbol(s)
        if sym and sym not in out:
            out.append(sym)
    return out


class WatchlistStore:
    """The user's watchlist, backed by ``QSettings`` (or an injected fake)."""

    def __init__(self, settings=None, seed: bool = True) -> None:
        self._s = settings if settings is not None else _default_settings()
        self._symbols: list[str] = []
        self.load()
        if seed:
            self.seed_if_first_run()
            self.seed_indices_once()

    # ── Access ──
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def __contains__(self, sym) -> bool:
        return normalize_symbol(sym) in self._symbols

    # ── Persistence ──
    def load(self) -> list[str]:
        self._symbols = _dedupe(_coerce_list(self._s.value(SYMBOLS_KEY)))
        return self.symbols()

    def save(self) -> None:
        self._s.setValue(SYMBOLS_KEY, list(self._symbols))

    # ── Mutation ──
    def add(self, raw) -> Optional[str]:
        """Normalize and append ``raw``. Returns the stored symbol, or ``None``
        if it was blank/junk (unresolvable) or already present."""
        sym = normalize_symbol(raw)
        if not sym or sym in self._symbols:
            return None
        self._symbols.append(sym)
        self.save()
        return sym

    def remove(self, raw) -> bool:
        sym = normalize_symbol(raw)
        if sym in self._symbols:
            self._symbols.remove(sym)
            self.save()
            return True
        return False

    def reorder(self, order) -> None:
        """Replace the list with ``order`` (normalized + de-duped), keeping only
        symbols already in the watchlist."""
        existing = set(self._symbols)
        new = [s for s in _dedupe(order) if s in existing]
        # Preserve any symbols the caller forgot to include, appended in old order.
        new += [s for s in self._symbols if s not in new]
        self._symbols = new
        self.save()

    # ── First-run seeding ──
    def seed_if_first_run(self) -> bool:
        """Seed the starters exactly once, on genuine first run. Never overwrites
        an existing (even empty) saved watchlist. Returns True if it seeded."""
        if self._s.value(INIT_KEY):
            return False
        self._s.setValue(INIT_KEY, "1")
        if self._symbols:
            return False  # respect a pre-existing list
        self._symbols = _dedupe(STARTERS)
        self.save()
        return True

    def seed_indices_once(self) -> bool:
        """One-time migration: ensure the market-context indices are present,
        prepended to the front. Runs once (flagged), so a user who later removes
        an index keeps it removed. Returns True if it added any."""
        if self._s.value(INDICES_KEY):
            return False
        self._s.setValue(INDICES_KEY, "1")
        missing = [s for s in INDEX_STARTERS if s not in self._symbols]
        if not missing:
            return False
        self._symbols = _dedupe(missing + self._symbols)
        self.save()
        return True
