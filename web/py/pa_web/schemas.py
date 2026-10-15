"""Request validation for the web demo. Caps keep a public demo cheap to run."""

from __future__ import annotations

import re
from datetime import date, timedelta

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_TICKERS = 20
MAX_YEARS = 15
MIN_DAYS = 180
EARLIEST = date(1995, 1, 1)
_SYMBOL = re.compile(r"[A-Z0-9.\-^=]{1,12}")


def _sym(s: str) -> str:
    s = s.strip().upper()
    if not _SYMBOL.fullmatch(s):
        raise ValueError(f"Invalid symbol {s!r}")
    return s


class Holding(BaseModel):
    ticker: str
    weight: float = Field(..., gt=0, le=100)

    @field_validator("ticker")
    @classmethod
    def _t(cls, v: str) -> str:
        return _sym(v)


class AnalyzeRequest(BaseModel):
    holdings: list[Holding] = Field(..., min_length=1, max_length=MAX_TICKERS)
    benchmark: str = "SPY"
    start_date: date
    end_date: date
    capital: float = Field(100_000.0, ge=1_000, le=100_000_000)

    @field_validator("benchmark")
    @classmethod
    def _b(cls, v: str) -> str:
        return _sym(v)

    @model_validator(mode="after")
    def _check(self) -> "AnalyzeRequest":
        tickers = [h.ticker for h in self.holdings]
        if len(set(tickers)) != len(tickers):
            raise ValueError("Each ticker may appear only once")
        today = date.today()
        if self.end_date > today:
            self.end_date = today
        if self.start_date < EARLIEST:
            raise ValueError(f"Start date must be on or after {EARLIEST.isoformat()}")
        if (self.end_date - self.start_date).days < MIN_DAYS:
            raise ValueError("Date range must be at least 6 months")
        if self.end_date - self.start_date > timedelta(days=round(365.25 * MAX_YEARS)):
            raise ValueError(f"Date range is limited to {MAX_YEARS} years in the demo")
        return self

    def normalized_weights(self) -> dict[str, float]:
        total = sum(h.weight for h in self.holdings)
        return {h.ticker: h.weight / total for h in self.holdings}

    def cache_key(self) -> str:
        w = self.normalized_weights()
        parts = [f"{t}:{w[t]:.6f}" for t in sorted(w)]
        return "|".join([*parts, self.benchmark, self.start_date.isoformat(),
                         self.end_date.isoformat(), f"{self.capital:.0f}"])


