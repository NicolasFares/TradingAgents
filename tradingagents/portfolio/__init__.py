"""Portfolio construction layer that wraps the per-ticker pipeline.

`TradingAgentsGraph.propagate(ticker, date)` produces one decision per
symbol; this module runs it across a basket and synthesises a single
allocation plan that respects basket-level constraints (cash floor, max
single position, max sector weight).
"""

from tradingagents.portfolio.constructor import (
    PortfolioConstructor,
    render_allocation_plan,
)
from tradingagents.portfolio.schemas import (
    AllocationPlan,
    PortfolioConstraints,
    TickerWeight,
)
from tradingagents.portfolio.validation import (
    BasketValidationError,
    TickerCandidate,
    TickerValidation,
    preflight_basket,
    validate_ticker,
)

__all__ = [
    "PortfolioConstructor",
    "AllocationPlan",
    "PortfolioConstraints",
    "TickerWeight",
    "render_allocation_plan",
    "BasketValidationError",
    "TickerCandidate",
    "TickerValidation",
    "preflight_basket",
    "validate_ticker",
]
