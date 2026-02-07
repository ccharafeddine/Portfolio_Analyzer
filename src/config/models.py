"""
Pydantic configuration models for Portfolio Analyzer v2.

Single source of truth — no more duplicate keys, no silent failures.
Validates everything at load time so analytics code can trust its inputs.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────────────────────


class BLView(BaseModel):
    """A single Black-Litterman view (absolute or relative)."""

    type: Literal["absolute", "relative"]
    confidence: Literal["low", "medium", "high"] = "medium"

    # Absolute view fields
    asset: Optional[str] = None
    q: float = 0.0

    # Relative view fields
    asset_long: Optional[str] = None
    asset_short: Optional[str] = None

    @model_validator(mode="after")
    def check_view_fields(self) -> "BLView":
        if self.type == "absolute" and not self.asset:
            raise ValueError("Absolute view requires 'asset'.")
        if self.type == "relative":
            if not self.asset_long or not self.asset_short:
                raise ValueError(
                    "Relative view requires both 'asset_long' and 'asset_short'."
                )
            if self.asset_long == self.asset_short:
                raise ValueError("asset_long and asset_short must differ.")
        return self


class BLConfig(BaseModel):
    """Black-Litterman model configuration."""

    enabled: bool = False
    tau: float = Field(0.05, gt=0, le=1.0)
    views: list[BLView] = Field(default_factory=list)


class CompletePortfolioConfig(BaseModel):
    """Controls the ORP ↔ risk-free mix."""

    y: float = Field(0.8, ge=0.0, le=1.0, description="Fraction in risky (ORP)")


# ──────────────────────────────────────────────────────────────
# Main config
# ──────────────────────────────────────────────────────────────


class PortfolioConfig(BaseModel):
    """
    Complete configuration for a portfolio analysis run.

    This replaces the old config.json with full validation.
    """

    # ── Universe ──
    tickers: list[str] = Field(..., min_length=1)
    weights: dict[str, float] = Field(...)
    benchmark: str = "SPY"

    # ── Date range ──
    start_date: date
    end_date: date

    # ── Capital & risk ──
    capital: float = Field(1_000_000.0, gt=0)
    risk_free_rate: float = Field(0.04, ge=0.0, le=0.25)

    # ── Optimization constraints ──
    short_sales: bool = False
    max_weight_bound: float = Field(1.0, ge=0.1, le=3.0)
    frontier_points: int = Field(50, ge=10, le=200)

    # ── Feature toggles ──
    include_orp: bool = True
    include_complete: bool = True
    use_dividends: bool = False

    # ── Sub-configs ──
    complete_portfolio: CompletePortfolioConfig = Field(
        default_factory=CompletePortfolioConfig
    )
    black_litterman: BLConfig = Field(default_factory=BLConfig)

    # ── Validators ──

    @field_validator("tickers", mode="before")
    @classmethod
    def uppercase_tickers(cls, v: list[str]) -> list[str]:
        return [t.strip().upper() for t in v if t.strip()]

    @field_validator("benchmark", mode="before")
    @classmethod
    def uppercase_benchmark(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("weights", mode="before")
    @classmethod
    def uppercase_weight_keys(cls, v: dict) -> dict:
        return {k.strip().upper(): float(w) for k, w in v.items()}

    @model_validator(mode="after")
    def validate_dates(self) -> "PortfolioConfig":
        if self.end_date <= self.start_date:
            raise ValueError(
                f"end_date ({self.end_date}) must be after start_date ({self.start_date})"
            )
        return self

    @model_validator(mode="after")
    def validate_weights(self) -> "PortfolioConfig":
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Weights sum to {total:.4f} — must be within 0.01 of 1.0"
            )

        # Every weighted ticker must be in tickers list
        missing = set(self.weights.keys()) - set(self.tickers)
        if missing:
            raise ValueError(
                f"Weights reference tickers not in universe: {missing}"
            )
        return self

    @model_validator(mode="after")
    def validate_bl_assets(self) -> "PortfolioConfig":
        if not self.black_litterman.enabled:
            return self
        universe = set(self.tickers)
        for i, view in enumerate(self.black_litterman.views):
            if view.type == "absolute" and view.asset not in universe:
                raise ValueError(
                    f"BL view {i}: asset '{view.asset}' not in ticker universe"
                )
            if view.type == "relative":
                if view.asset_long not in universe:
                    raise ValueError(
                        f"BL view {i}: asset_long '{view.asset_long}' not in universe"
                    )
                if view.asset_short not in universe:
                    raise ValueError(
                        f"BL view {i}: asset_short '{view.asset_short}' not in universe"
                    )
        return self

    # ── Computed properties ──

    @property
    def allocation_bounds(self) -> tuple[float, float]:
        lo = -self.max_weight_bound if self.short_sales else 0.0
        return (lo, self.max_weight_bound)

    @property
    def start_str(self) -> str:
        return self.start_date.isoformat()

    @property
    def end_str(self) -> str:
        return self.end_date.isoformat()

    # ── IO ──

    def save(self, path: str | Path) -> None:
        """Write config to JSON."""
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "PortfolioConfig":
        """Load and validate config from JSON."""
        with open(path) as f:
            raw = json.load(f)
        return cls(**raw)

    @classmethod
    def from_legacy(cls, path: str | Path) -> "PortfolioConfig":
        """
        Load from an old-format config.json (with duplicate keys, etc.)
        and normalize into the new schema.
        """
        with open(path) as f:
            old = json.load(f)

        ap = old.get("active_portfolio", {}) or {}

        # Resolve the many possible date key locations
        start = (
            old.get("start")
            or old.get("start_date")
            or ap.get("start_date")
            or "2020-01-01"
        )
        end = (
            old.get("end")
            or old.get("end_date")
            or ap.get("end_date")
            or "2025-12-31"
        )

        tickers = old.get("tickers", ap.get("tickers", []))
        weights = ap.get("weights", old.get("weights", {}))
        benchmark = old.get("benchmark") or old.get("passive_benchmark", "SPY")
        capital = ap.get("capital", old.get("initial_capital", 1_000_000))

        # Normalize bounds
        bounds = old.get("max_allocation_bounds", [0.0, 1.0])
        max_bound = bounds[1] if isinstance(bounds, list) and len(bounds) >= 2 else 1.0

        # BL config
        bl_raw = old.get("black_litterman", {}) or {}
        bl_config = BLConfig(
            enabled=bool(bl_raw.get("enabled", False)),
            tau=float(bl_raw.get("tau", 0.05)),
            views=[BLView(**v) for v in (bl_raw.get("views") or [])],
        )

        # Complete portfolio
        cp_raw = old.get("complete_portfolio", {}) or {}
        y_val = cp_raw.get("y", old.get("y_cp", 0.8))

        return cls(
            tickers=tickers,
            weights=weights,
            benchmark=benchmark,
            start_date=start,
            end_date=end,
            capital=capital,
            risk_free_rate=old.get("risk_free_rate", 0.04),
            short_sales=bool(old.get("short_sales", False)),
            max_weight_bound=max_bound,
            frontier_points=int(old.get("frontier_points", 50)),
            include_orp=bool(old.get("include_orp", True)),
            include_complete=bool(old.get("include_complete", True)),
            use_dividends=bool(ap.get("use_dividends", False)),
            complete_portfolio=CompletePortfolioConfig(y=y_val),
            black_litterman=bl_config,
        )
