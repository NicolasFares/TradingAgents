"""Prompt construction for the basket-level synthesis call."""

from __future__ import annotations

from tradingagents.portfolio.schemas import PortfolioConstraints


def build_synthesis_prompt(
    decisions: dict[str, str],
    holdings: dict[str, float],
    constraints: PortfolioConstraints,
    trade_date: str,
) -> str:
    """Render the full basket-synthesis prompt.

    ``decisions`` maps ticker -> the markdown string that the per-ticker
    Portfolio Manager produced (``final_state["final_trade_decision"]``).
    The synthesiser reads these verbatim — no re-summarisation — so it
    sees the same evidence the per-ticker pipeline reached its rating on.
    """
    per_ticker_block = "\n\n".join(
        f"### {ticker}\n{markdown.strip()}"
        for ticker, markdown in decisions.items()
    )

    if holdings:
        holdings_str = ", ".join(f"{k}: {v:.1%}" for k, v in holdings.items())
    else:
        holdings_str = "all cash (no existing positions)"

    sector_lines = (
        "\n".join(f"  - {t}: {s}" for t, s in constraints.sector_map.items())
        if constraints.sector_map
        else "  (no sector mapping supplied)"
    )

    return f"""You are constructing a portfolio allocation across {len(decisions)} analysed names
as of {trade_date}, given the per-ticker analyst conclusions below.

## Per-ticker analyst conclusions

Each section is the final Portfolio Manager output for that single name,
rendered verbatim. Treat these as the only source of view on each ticker —
do not invent fundamentals or news that the analysts did not surface.

{per_ticker_block}

## Current portfolio

{holdings_str}

## Constraints

- Cash floor: at least {constraints.cash_floor:.1%} of NAV must remain in cash.
- Max single position: no single ticker may exceed {constraints.max_single_position:.1%} of NAV.
- Max sector weight: combined weight of tickers in the same sector may not exceed {constraints.max_sector_weight:.1%}.
- Sector mapping:
{sector_lines}

## Your task

Produce an AllocationPlan as a JSON object with EXACTLY these top-level keys:
``allocations``, ``cash_weight``, ``portfolio_thesis``, ``risk_notes``, ``rejected_names``.

Each entry inside ``allocations`` is an object with EXACTLY these keys:
``ticker``, ``target_weight``, ``action``, ``rationale``.

Field semantics:

- ``allocations`` — one entry per ticker you choose to hold. ``target_weight``
  is the fraction of NAV (0.0–1.0). ``action`` is one of buy, add, hold,
  trim, sell (lower-case). ``rationale`` is one to three sentences citing the
  relevant per-ticker thesis.
- ``cash_weight`` — fraction kept in cash. Sum of all ``target_weight`` values
  plus ``cash_weight`` must equal 1.0.
- ``portfolio_thesis`` — basket-level view explaining how the weights express
  your macro and risk stance (3-6 sentences).
- ``risk_notes`` — concentration, correlation, and macro risks that would
  invalidate the plan (2-4 sentences).
- ``rejected_names`` — list of strings, each formatted ``"TICKER: one-line reason"``.

Use the exact key names above. Do not substitute ``weight`` for ``target_weight``
or ``reason`` for ``rationale``. Respect every constraint. If a per-ticker thesis
is weak or contradictory, reflect that with a smaller weight or exclusion rather
than overcommitting.
"""
