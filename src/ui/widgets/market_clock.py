"""Market session clock + opening/closing bell for the Live Market Watch top bar.

Shows a colored status dot, a label (Open / Pre-market / After hours / Closed) and
a countdown to the next open or close, ticking every second in ET. On the
transition into and out of the regular session it plays a short **synthesized**
chime (rising on the open, falling on the close) — no bundled audio asset, so
there's nothing to license. A speaker button mutes it (persisted).
"""

from __future__ import annotations

import math
import struct
import wave
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QWidget

from .. import paths, theme
from src.data import market_session as session

_MUTE_KEY = "market_bell_muted"


def _sounds_dir():
    d = paths.data_dir() / "sounds"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_chime(path, notes) -> None:
    """Write a short bell-ish chime WAV. ``notes`` is a list of (freq_hz, seconds);
    each note is a decaying fundamental plus two faster-decaying harmonics."""
    rate = 44100
    frames = bytearray()
    for freq, dur in notes:
        n = int(rate * dur)
        for i in range(n):
            t = i / rate
            env = math.exp(-3.2 * t)
            s = (math.sin(2 * math.pi * freq * t)
                 + 0.5 * math.sin(2 * math.pi * 2 * freq * t) * math.exp(-5 * t)
                 + 0.25 * math.sin(2 * math.pi * 3 * freq * t) * math.exp(-7 * t))
            val = int(max(-1.0, min(1.0, 0.32 * env * s)) * 32767)
            frames += struct.pack("<h", val)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


def _ensure_chimes():
    d = _sounds_dir()
    opening = d / "bell_open.wav"
    closing = d / "bell_close.wav"
    if not opening.exists():
        _write_chime(opening, [(587.33, 0.35), (880.0, 0.85)])   # rising D5→A5
    if not closing.exists():
        _write_chime(closing, [(880.0, 0.35), (587.33, 0.85)])   # falling A5→D5
    return opening, closing


class MarketClock(QWidget):
    """Live session status + countdown, with an open/close bell."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._prev_state: Optional[str] = None
        self._open_snd = self._close_snd = None
        self._init_audio()

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._dot = QLabel("●")
        self._label = QLabel("")
        self._mute = QToolButton()
        self._mute.setCheckable(True)
        self._mute.setCursor(Qt.PointingHandCursor)
        self._mute.setAutoRaise(True)
        self._mute.setChecked(self._muted())
        self._mute.toggled.connect(self._on_mute)
        row.addWidget(self._dot)
        row.addWidget(self._label)
        row.addWidget(self._mute)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._tick(ring=False)   # seed state without ringing on launch
        self.retheme()

    # ── Audio ──
    def _init_audio(self) -> None:
        try:
            from PySide6.QtMultimedia import QSoundEffect

            opening, closing = _ensure_chimes()
            self._open_snd = QSoundEffect(self)
            self._open_snd.setSource(QUrl.fromLocalFile(str(opening)))
            self._open_snd.setVolume(0.6)
            self._close_snd = QSoundEffect(self)
            self._close_snd.setSource(QUrl.fromLocalFile(str(closing)))
            self._close_snd.setVolume(0.6)
        except Exception:
            self._open_snd = self._close_snd = None  # fall back to winsound / silent

    def _play(self, opening: bool) -> None:
        if self._muted():
            return
        eff = self._open_snd if opening else self._close_snd
        if eff is not None:
            eff.play()
            return
        try:  # Windows fallback
            import winsound

            p, q = _ensure_chimes()
            winsound.PlaySound(str(p if opening else q),
                               winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            pass

    # ── Mute state (QSettings) ──
    def _settings(self):
        from PySide6.QtCore import QSettings

        from ..settings import APP_NAME, ORG_NAME

        return QSettings(ORG_NAME, APP_NAME)

    def _muted(self) -> bool:
        return bool(self._settings().value(_MUTE_KEY, False, type=bool))

    def _on_mute(self, checked: bool) -> None:
        self._settings().setValue(_MUTE_KEY, bool(checked))
        self._update_mute_icon()

    def _update_mute_icon(self) -> None:
        self._mute.setText("🔇" if self._mute.isChecked() else "🔔")
        self._mute.setToolTip("Bell muted — click to unmute" if self._mute.isChecked()
                              else "Opening/closing bell on — click to mute")

    # ── Tick ──
    def _tick(self, ring: bool = True) -> None:
        st = session.status()
        # Bell on the transition into / out of the regular session.
        if ring and self._prev_state is not None and st.state != self._prev_state:
            if st.state == session.OPEN_STATE:
                self._play(opening=True)
            elif self._prev_state == session.OPEN_STATE:
                self._play(opening=False)
        self._prev_state = st.state

        if st.state == session.OPEN_STATE:
            target, verb = st.next_change, "closes"
        elif st.state == session.PRE:
            target, verb = st.next_change, "opens"
        else:
            target, verb = session.next_regular_open(), "opens"
        from datetime import datetime

        secs = max(0, int((target - datetime.now(session.ET)).total_seconds()))
        self._label.setText(f"{st.label} · {verb} in {_fmt_delta(secs)}")
        self._colorize(st.state)

    def _colorize(self, state: str) -> None:
        t = theme.ACTIVE
        color = {
            session.OPEN_STATE: t.green,
            session.PRE: t.accent,
            session.AFTER: t.accent,
            session.CLOSED: t.text_muted,
        }.get(state, t.text_muted)
        self._dot.setStyleSheet(f"color:{color}; font-size:{t.base_pt}px;")

    def retheme(self) -> None:
        t = theme.ACTIVE
        self._label.setStyleSheet(f"color:{t.text_slate}; font-size:{t.base_pt - 1}px;")
        self._mute.setStyleSheet("QToolButton{border:none;background:transparent;}")
        self._update_mute_icon()
        if self._prev_state is not None:
            self._colorize(self._prev_state)

    def shutdown(self) -> None:
        self._timer.stop()


def _fmt_delta(secs: int) -> str:
    h, rem = divmod(secs, 3600)
    m, _ = divmod(rem, 60)
    if h >= 24:
        return f"{h // 24}d {h % 24}h"
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m"
    return f"{secs}s"
