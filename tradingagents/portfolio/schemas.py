"""Pydantic schemas for basket-level allocation output.

Mirrors the conventions in ``tradingagents/agents/schemas.py``: Literal
enums for actions, ``Field(description=...)`` doubling as the model's
output instructions, and a ``render_*`` companion in
``constructor.py`` that turns the parsed instance back into the
markdown shape the rest of the system reads.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


TickerAction = Literal["buy", "add", "hold", "trim", "sell"]


# Tolerant config: open-weights models (e.g. gpt-oss via Ollama's json_schema mode)
# treat the schema as a strong hint rather than a hard constraint and will sometimes
# emit synonyms (`weight` for `target_weight`, `reason` for `rationale`) or omit
# fields the prompt didn't reinforce. `extra="ignore"` plus `populate_by_name=True`
# plus `validation_alias=AliasChoices(...)` keeps the validation pass robust to
# those drifts without weakening the contract for well-behaved providers.
_TOLERANT = ConfigDict(extra="ignore", populate_by_name=True)


class TickerWeight(BaseModel):
    model_config = _TOLERANT

    ticker: str = Field(
        description="Ticker symbol exactly as supplied in the input basket.",
        validation_alias=AliasChoices("ticker", "symbol", "name"),
    )
    target_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Target fraction of net asset value, between 0.0 and 1.0.",
        validation_alias=AliasChoices("target_weight", "weight", "allocation"),
    )
    action: TickerAction = Field(
        default="hold",
        description=(
            "Direction implied by moving from the current holding to target_weight: "
            "buy (new position from zero), add (increase existing), hold (no change), "
            "trim (reduce existing), sell (exit to zero)."
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "One to three sentences citing the per-ticker analyst thesis that "
            "justifies this weight. Do not invent views the analysts did not take."
        ),
        validation_alias=AliasChoices("rationale", "reason", "justification"),
    )


class PortfolioConstraints(BaseModel):
    cash_floor: float = Field(default=0.05, ge=0.0, le=1.0)
    max_single_position: float = Field(default=0.20, ge=0.0, le=1.0)
    max_sector_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    sector_map: dict[str, str] = Field(
        default_factory=dict,
        description="Optional ticker -> sector mapping used to enforce max_sector_weight.",
    )


class AllocationPlan(BaseModel):
    model_config = _TOLERANT

    allocations: list[TickerWeight] = Field(
        description=(
            "One entry per ticker the synthesiser allocates to. Omit tickers that "
            "are excluded entirely; list them in rejected_names instead."
        ),
    )
    cash_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of NAV kept in cash. Sum of allocation weights + cash_weight must equal 1.0.",
    )
    portfolio_thesis: str = Field(
        description=(
            "One paragraph (3-6 sentences) explaining the basket-level view: "
            "how the chosen weights express the synthesiser's macro and risk stance, "
            "anchored in the per-ticker theses."
        ),
    )
    risk_notes: str = Field(
        description=(
            "Concentration risks, cross-ticker correlations, and macro factors that "
            "would invalidate the plan. Two to four sentences."
        ),
    )
    rejected_names: list[str] = Field(
        default_factory=list,
        description=(
            "Tickers excluded from the basket entirely. Each entry is a string of "
            "the form 'TICKER: one-line reason' so the user can audit exclusions."
        ),
    )
