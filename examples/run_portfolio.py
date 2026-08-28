"""Run the portfolio constructor on a fixed basket and dump results to disk.

Usage:
    .venv/bin/python -u examples/run_portfolio.py

Uses the default .env-driven config (Ollama on the DGX Spark via SSH tunnel).
Writes per-ticker decisions, basket plan (JSON + markdown), and a terminal
summary to ``~/.tradingagents/logs/_portfolio/<today>/``.

Progress markers printed to stdout (line-prefixed for easy grep / monitor):
    [PROGRESS] starting <ticker> (<i>/<n>)
    [PER_TICKER_DONE] <ticker>
    [PER_TICKER_FAILED] <ticker>: <error>
    [SYNTHESIS_START]
    [SYNTHESIS_DONE]
    [WRITE_DONE] <path>
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import sys
import time
import traceback
from pathlib import Path

# Surface logger.warning(...) from the constructor so silent failures are visible.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Trigger load_dotenv so TRADINGAGENTS_* env vars from the repo-root .env
# are applied before DEFAULT_CONFIG is built.
import tradingagents  # noqa: F401

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.portfolio import (
    AllocationPlan,
    BasketValidationError,
    PortfolioConstraints,
    PortfolioConstructor,
    preflight_basket,
    render_allocation_plan,
)


USER_BASKET = ["VUAA", "EUDF", "DLN", "CGL.C.TO", "ZST.TO", "BTCX.B.TO", "VFV.TO"]


def _output_dir(trade_date: str) -> Path:
    root = Path(DEFAULT_CONFIG["results_dir"]).expanduser() / "_portfolio" / trade_date
    (root / "per_ticker").mkdir(parents=True, exist_ok=True)
    return root


def _summary_table(plan: AllocationPlan) -> str:
    lines = ["Ticker            Weight   Action"]
    for a in plan.allocations:
        lines.append(f"{a.ticker:16s} {a.target_weight:>6.1%}   {a.action}")
    lines.append(f"{'CASH':16s} {plan.cash_weight:>6.1%}   hold")
    return "\n".join(lines)


def main() -> int:
    trade_date = _dt.date.today().isoformat()
    out_dir = _output_dir(trade_date)

    print(f"[INFO] trade_date={trade_date}")
    print(f"[INFO] basket={USER_BASKET}")
    print(f"[INFO] llm_provider={DEFAULT_CONFIG['llm_provider']} "
          f"deep={DEFAULT_CONFIG['deep_think_llm']} "
          f"quick={DEFAULT_CONFIG['quick_think_llm']} "
          f"backend={DEFAULT_CONFIG.get('backend_url')}")
    print(f"[INFO] output={out_dir}")

    # Pre-flight: validate every ticker against yfinance before committing to
    # the 60+ minute per-ticker phase. Interactive when run from a TTY; in
    # background / scripted runs, abort cleanly if anything fails to resolve.
    is_tty = sys.stdin.isatty()
    try:
        basket = preflight_basket(
            USER_BASKET,
            interactive=is_tty,
            on_unresolved="abort",
        )
    except BasketValidationError as exc:
        print(f"[ERROR] preflight aborted: {exc}")
        print("[ERROR] unresolved tickers:", exc.unresolved)
        print("[HINT] re-run interactively (TTY) to pick suggested substitutes, "
              "or fix the symbols in USER_BASKET and try again.")
        return 1

    if not basket:
        print("[ERROR] preflight produced an empty basket; nothing to analyse.")
        return 1

    ctor = PortfolioConstructor()
    print(f"[INFO] PortfolioConstructor ready; starting per-ticker phase on {len(basket)} tickers")

    start = time.time()
    n = len(basket)
    completed = 0

    def on_done(ticker: str, result) -> None:
        nonlocal completed
        completed += 1
        per_ticker_dir = out_dir / "per_ticker"
        if isinstance(result, Exception):
            tb = "".join(traceback.format_exception(type(result), result, result.__traceback__))
            (per_ticker_dir / f"{ticker}.error.txt").write_text(tb, encoding="utf-8")
            print(f"[PER_TICKER_FAILED] {ticker} ({completed}/{n}): "
                  f"{type(result).__name__}: {result}", flush=True)
        else:
            (per_ticker_dir / f"{ticker}.md").write_text(result, encoding="utf-8")
            print(f"[PER_TICKER_DONE] {ticker} ({completed}/{n}) "
                  f"elapsed={time.time() - start:.0f}s", flush=True)

    def on_synth_start(n_ok: int) -> None:
        print(f"[SYNTHESIS_START] {n_ok}/{n} tickers succeeded — synthesising basket plan", flush=True)

    constraints = PortfolioConstraints()
    per_ticker, plan = ctor.run(
        tickers=basket,
        trade_date=trade_date,
        current_holdings=None,
        constraints=constraints,
        on_ticker_complete=on_done,
        on_synthesis_start=on_synth_start,
    )

    successful = {t: md for t, md in per_ticker.items() if not isinstance(md, Exception)}
    if not successful:
        print("[ERROR] No tickers succeeded; cannot synthesise plan.")
        return 1

    if plan is None:
        print("[SYNTHESIS_FAILED]")
        print("[ERROR] Synthesis returned no AllocationPlan (structured + free-text both failed).")
        (out_dir / "synthesis_failed.txt").write_text(
            "Structured synthesis and the free-text fallback both failed. "
            "See per_ticker/*.md for the inputs that were given to the synthesiser.",
            encoding="utf-8",
        )
        return 2

    print("[SYNTHESIS_DONE]")

    plan_md = render_allocation_plan(plan)
    plan_json = json.dumps(plan.model_dump(mode="json"), indent=2)
    summary = _summary_table(plan)

    (out_dir / "plan.md").write_text(plan_md, encoding="utf-8")
    (out_dir / "plan.json").write_text(plan_json, encoding="utf-8")
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")

    print(f"[WRITE_DONE] {out_dir / 'plan.md'}")
    print(f"[WRITE_DONE] {out_dir / 'plan.json'}")
    print(f"[WRITE_DONE] {out_dir / 'summary.txt'}")

    print()
    print("===== FINAL ALLOCATION =====")
    print(summary)
    print("============================")
    print(f"[INFO] total elapsed: {time.time() - start:.0f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
