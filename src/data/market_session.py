"""U.S. equity market (NYSE/Nasdaq) session status — Qt-free and timezone-correct.

Reports whether the regular session is open, in pre-market, after-hours, or
closed, and when the next transition happens, in America/New_York regardless of
the user's local timezone. Accounts for weekends and the NYSE holiday calendar
(fixed + nth-weekday holidays, plus Good Friday), with the standard
Saturday→Friday / Sunday→Monday observance shift.

Early-close half-days (1:00pm on a few days like the day after Thanksgiving) are
treated as normal full sessions — a minor simplification for a status indicator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

PRE_OPEN = time(4, 0)     # pre-market begins
OPEN = time(9, 30)        # regular session open
CLOSE = time(16, 0)       # regular session close
AFTER_END = time(20, 0)   # after-hours ends

# Session states.
CLOSED = "closed"
PRE = "pre"
OPEN_STATE = "open"
AFTER = "after"

_LABELS = {
    CLOSED: "Market closed",
    PRE: "Pre-market",
    OPEN_STATE: "Market open",
    AFTER: "After hours",
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The ``n``-th ``weekday`` (Mon=0) of ``month``; ``n=-1`` = last."""
    if n > 0:
        d = date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + timedelta(days=offset + (n - 1) * 7)
    # last weekday of the month
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    d = nxt - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = ((h + m - 7 * n + 114) % 31) + 1
    return date(year, month, day)


def _observed(d: date) -> date:
    """NYSE observance: a Saturday holiday shifts to Friday, Sunday to Monday."""
    if d.weekday() == 5:      # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:      # Sunday
        return d + timedelta(days=1)
    return d


def nyse_holidays(year: int) -> set[date]:
    """Observed NYSE market holidays for ``year``."""
    days = {
        _observed(date(year, 1, 1)),                     # New Year's Day
        _nth_weekday(year, 1, 0, 3),                     # MLK Jr. Day (3rd Mon Jan)
        _nth_weekday(year, 2, 0, 3),                     # Washington's Birthday (3rd Mon Feb)
        _easter(year) - timedelta(days=2),               # Good Friday
        _nth_weekday(year, 5, 0, -1),                    # Memorial Day (last Mon May)
        _nth_weekday(year, 9, 0, 1),                     # Labor Day (1st Mon Sep)
        _nth_weekday(year, 11, 3, 4),                    # Thanksgiving (4th Thu Nov)
        _observed(date(year, 12, 25)),                   # Christmas
        _observed(date(year, 7, 4)),                     # Independence Day
    }
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))           # Juneteenth (from 2022)
    return days


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def _next_trading_day(d: date) -> date:
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


@dataclass
class SessionStatus:
    state: str                    # CLOSED / PRE / OPEN_STATE / AFTER
    label: str                    # human label
    next_change: datetime         # tz-aware ET datetime of the next transition
    is_open: bool                 # regular session open

    @property
    def seconds_to_change(self) -> int:
        return max(0, int((self.next_change - datetime.now(ET)).total_seconds()))


def _at(d: date, t: time) -> datetime:
    return datetime.combine(d, t, tzinfo=ET)


def next_regular_open(now: Optional[datetime] = None) -> datetime:
    """The next regular-session open (09:30 ET) at or after ``now``."""
    now_et = (now.astimezone(ET) if now is not None else datetime.now(ET))
    today = now_et.date()
    if is_trading_day(today) and now_et.timetz().replace(tzinfo=None) < OPEN:
        return _at(today, OPEN)
    return _at(_next_trading_day(today), OPEN)


def status(now: Optional[datetime] = None) -> SessionStatus:
    """Current session status in ET. ``now`` may be any tz-aware datetime (or None
    for the real clock); it is converted to Eastern time."""
    now_et = (now.astimezone(ET) if now is not None else datetime.now(ET))
    today = now_et.date()
    t = now_et.timetz().replace(tzinfo=None)

    if is_trading_day(today):
        if t < PRE_OPEN:
            return SessionStatus(CLOSED, _LABELS[CLOSED], _at(today, PRE_OPEN), False)
        if t < OPEN:
            return SessionStatus(PRE, _LABELS[PRE], _at(today, OPEN), False)
        if t < CLOSE:
            return SessionStatus(OPEN_STATE, _LABELS[OPEN_STATE], _at(today, CLOSE), True)
        if t < AFTER_END:
            return SessionStatus(AFTER, _LABELS[AFTER], _at(today, AFTER_END), False)
    # Closed: next change is pre-market open of the next trading day.
    nxt = today if (is_trading_day(today) and t < PRE_OPEN) else _next_trading_day(today)
    return SessionStatus(CLOSED, _LABELS[CLOSED], _at(nxt, PRE_OPEN), False)
