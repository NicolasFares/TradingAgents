"""Portfolio construction orchestrator.

Wraps the per-ticker pipeline (``TradingAgentsGraph.propagate``) in a loop
and adds one final structured-output LLM call that synthesises a single
``AllocationPlan`` across the basket. The synthesis reuses the same
``bind_structured`` / fallback pattern as the Research Manager, Trader,
and per-ticker Portfolio Manager (see ``tradingagents/agents/utils/structured.py``).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from tradingagents.agents.utils.structured import bind_structured
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.portfolio.prompts import build_synthesis_prompt
from tradingagents.portfolio.schemas import (
    AllocationPlan,
    PortfolioConstraints,
    TickerWeight,
)


logger = logging.getLogger(__name__)


OnTickerComplete = Callable[[str, "str | Exception"], None]
OnSynthesisStart = Callable[[int], None]  # arg: count of successful tickers


class PortfolioConstructor:
    """Run the per-ticker pipeline across a basket and synthesise an allocation."""

    def __init__(
        self,
        ta_graph: Optional[TradingAgentsGraph] = None,
        config: Optional[dict] = None,
    ):
        self.config = (config or DEFAULT_CONFIG).copy()
        self.ta = ta_graph or TradingAgentsGraph(debug=False, config=self.config)

        synth_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
        )
        self._raw_synth = synth_client.get_llm()
        self._structured_synth = bind_structured(
            self._raw_synth, AllocationPlan, "Portfolio Constructor"
        )

    def run(
        self,
        tickers: list[str],
        trade_date: str,
        current_holdings: Optional[dict[str, float]] = None,
        constraints: Optional[PortfolioConstraints] = None,
        on_ticker_complete: Optional[OnTickerComplete] = None,
        on_synthesis_start: Optional[OnSynthesisStart] = None,
    ) -> tuple[dict[str, "str | Exception"], Optional[AllocationPlan]]:
        """Analyse each ticker, then synthesise a basket-level AllocationPlan.

        Returns ``(per_ticker, plan)`` where ``per_ticker`` maps each input
        ticker either to its rendered Portfolio Manager markdown (success)
        or to the Exception raised during ``propagate`` (failure), and
        ``plan`` is the synthesised ``AllocationPlan`` — or ``None`` if no
        ticker succeeded or the synthesis itself failed.
        """
        constraints = constraints or PortfolioConstraints()
        per_ticker: dict[str, "str | Exception"] = {}

        for ticker in tickers:
            try:
                final_state, _ = self.ta.propagate(ticker, trade_date)
                per_ticker[ticker] = final_state["final_trade_decision"]
            except Exception as exc:
                logger.warning("Per-ticker analysis failed for %s: %s", ticker, exc)
                per_ticker[ticker] = exc
            if on_ticker_complete is not None:
                on_ticker_complete(ticker, per_ticker[ticker])

        successful = {
            t: md for t, md in per_ticker.items() if not isinstance(md, Exception)
        }
        if not successful:
            logger.warning("No tickers succeeded; skipping basket synthesis.")
            return per_ticker, None

        if on_synthesis_start is not None:
            on_synthesis_start(len(successful))

        prompt = build_synthesis_prompt(
            decisions=successful,
            holdings=current_holdings or {},
            constraints=constraints,
            trade_date=trade_date,
        )

        plan = _invoke_synthesis(
            self._structured_synth, self._raw_synth, prompt, AllocationPlan
        )
        return per_ticker, plan


def _invoke_synthesis(
    structured_llm,
    plain_llm,
    prompt: str,
    schema: type[AllocationPlan],
) -> Optional[AllocationPlan]:
    """Run the structured synthesis call; fall back to free-text parse on failure.

    Unlike ``invoke_structured_or_freetext`` in ``agents/utils/structured.py``
    (which returns a markdown string), this returns the typed
    ``AllocationPlan`` so the runner can write both ``plan.json`` and
    ``plan.md`` cleanly. If the structured call fails, we try one
    free-text invocation and attempt JSON parse; if that also fails we
    return ``None`` and let the caller surface the raw response.
    """
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            if result is not None:
                return result
            # Some langchain providers silently return None when the model's
            # JSON fails internal validation instead of raising. Treat that
            # as a structured failure and fall through to the free-text path.
            logger.warning(
                "Portfolio Constructor: structured synthesis returned None "
                "(model JSON likely failed internal validation); "
                "retrying once as free text"
            )
        except Exception as exc:
            logger.warning(
                "Portfolio Constructor: structured synthesis failed (%s); "
                "retrying once as free text",
                exc,
            )

    try:
        response = plain_llm.invoke(prompt)
        raw = getattr(response, "content", str(response))
        return schema.model_validate_json(_extract_json_block(raw))
    except Exception as exc:
        logger.warning("Portfolio Constructor: free-text synthesis also failed: %s", exc)
        return None


def _extract_json_block(text: str) -> str:
    """Pull the first {...} JSON block out of a free-text LLM response."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def render_allocation_plan(plan: AllocationPlan) -> str:
    """Render an AllocationPlan back to markdown for logging and display."""
    lines: list[str] = ["# Portfolio Allocation Plan", ""]

    lines.append("## Allocations")
    lines.append("")
    lines.append("| Ticker | Weight | Action | Rationale |")
    lines.append("|---|---:|---|---|")
    for alloc in plan.allocations:
        lines.append(
            f"| {alloc.ticker} | {alloc.target_weight:.1%} | {alloc.action} | "
            f"{alloc.rationale.replace(chr(10), ' ')} |"
        )
    lines.append(f"| **Cash** | {plan.cash_weight:.1%} | hold | reserve |")

    lines.extend(["", "## Portfolio Thesis", "", plan.portfolio_thesis])
    lines.extend(["", "## Risk Notes", "", plan.risk_notes])

    if plan.rejected_names:
        lines.extend(["", "## Rejected"])
        for item in plan.rejected_names:
            lines.append(f"- {item}")

    return "\n".join(lines)
