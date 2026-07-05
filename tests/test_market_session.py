"""NYSE market-session logic (Qt-free): holidays + open/pre/after/closed states."""

from datetime import date, datetime, timezone

from src.data import market_session as ms
from src.data.market_session import ET


def test_holidays_2026():
    hol = ms.nyse_holidays(2026)
    assert date(2026, 1, 1) in hol          # New Year's Day (Thu)
    assert date(2026, 1, 19) in hol         # MLK (3rd Mon Jan)
    assert date(2026, 2, 16) in hol         # Washington's Birthday (3rd Mon Feb)
    assert date(2026, 4, 3) in hol          # Good Friday (Easter Apr 5)
    assert date(2026, 5, 25) in hol         # Memorial Day (last Mon May)
    assert date(2026, 6, 19) in hol         # Juneteenth
    assert date(2026, 7, 3) in hol          # Independence Day observed (Jul 4 = Sat)
    assert date(2026, 9, 7) in hol          # Labor Day (1st Mon Sep)
    assert date(2026, 11, 26) in hol        # Thanksgiving (4th Thu Nov)
    assert date(2026, 12, 25) in hol        # Christmas (Fri)


def test_is_trading_day():
    assert ms.is_trading_day(date(2026, 3, 4)) is True       # Wednesday
    assert ms.is_trading_day(date(2026, 3, 7)) is False      # Saturday
    assert ms.is_trading_day(date(2026, 3, 8)) is False      # Sunday
    assert ms.is_trading_day(date(2026, 12, 25)) is False    # holiday


def _et(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=ET)


def test_states_across_a_trading_day():
    day = (2026, 3, 4)  # Wednesday

    s = ms.status(_et(*day, 3))
    assert s.state == ms.CLOSED and not s.is_open
    assert s.next_change.hour == 4                # opens pre-market at 4:00

    s = ms.status(_et(*day, 8))
    assert s.state == ms.PRE and s.next_change.hour == 9 and s.next_change.minute == 30

    s = ms.status(_et(*day, 10))
    assert s.state == ms.OPEN_STATE and s.is_open
    assert s.next_change.hour == 16               # closes at 16:00

    s = ms.status(_et(*day, 17))
    assert s.state == ms.AFTER and not s.is_open and s.next_change.hour == 20

    s = ms.status(_et(*day, 21))
    assert s.state == ms.CLOSED
    assert s.next_change.date() == date(2026, 3, 5)  # next trading day pre-market


def test_weekend_next_change_is_monday():
    s = ms.status(_et(2026, 3, 7, 12))               # Saturday
    assert s.state == ms.CLOSED
    assert s.next_change.date() == date(2026, 3, 9)  # Monday
    assert s.next_change.hour == 4


def test_holiday_is_closed():
    assert ms.status(_et(2026, 12, 25, 11)).state == ms.CLOSED


def test_next_regular_open():
    # Before the open on a trading day → today 9:30.
    o = ms.next_regular_open(_et(2026, 3, 4, 8))
    assert o.date() == date(2026, 3, 4) and o.hour == 9 and o.minute == 30
    # After the close → next trading day 9:30.
    o = ms.next_regular_open(_et(2026, 3, 4, 17))
    assert o.date() == date(2026, 3, 5) and o.hour == 9
    # On a holiday/weekend → the following trading day.
    o = ms.next_regular_open(_et(2026, 12, 25, 11))
    assert o.date() == date(2026, 12, 28)   # Christmas Fri → Monday


def test_accepts_utc_and_converts_to_eastern():
    # 15:00 UTC on 2026-03-04 is 10:00 ET (EST, before DST) → open.
    s = ms.status(datetime(2026, 3, 4, 15, 0, tzinfo=timezone.utc))
    assert s.state == ms.OPEN_STATE
