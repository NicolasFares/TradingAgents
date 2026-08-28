"""Re-run only the basket-synthesis step against cached per-ticker decisions.

Reads ~/.tradingagents/logs/_portfolio/<date>/per_ticker/*.md (the rendered
Portfolio Manager outputs from a previous run) and invokes the structured
synthesis call. Bypasses the ~64-minute per-ticker phase so we can iterate
on the synthesis prompt / schema without redoing analyst work.

Usage:
    .venv/bin/python -u examples/synthesize_only.py [YYYY-MM-DD]
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import tradingagents  # noqa: F401  - triggers load_dotenv

from tradingagents.agents.utils.structured import bind_structured
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.portfolio.prompts import build_synthesis_prompt
from tradingagents.portfolio.schemas import AllocationPlan, PortfolioConstraints
from tradingagents.portfolio.constructor import render_allocation_plan

# Surface logger.warning(...) from constructor + here
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main(date: str) -> int:
    out_dir = Path(DEFAULT_CONFIG["results_dir"]).expanduser() / "_portfolio" / date
    pt_dir = out_dir / "per_ticker"

    md_files = sorted(p for p in pt_dir.glob("*.md"))
    if not md_files:
        print(f"[ERROR] no per_ticker/*.md files found under {pt_dir}")
        return 1

    decisions = {p.stem: p.read_text(encoding="utf-8") for p in md_files}
    print(f"[INFO] loaded {len(decisions)} cached decisions: {list(decisions)}")

    constraints = PortfolioConstraints()
    prompt = build_synthesis_prompt(
        decisions=decisions, holdings={}, constraints=constraints, trade_date=date
    )
    print(f"[INFO] prompt length: {len(prompt)} chars")

    llm_client = create_llm_client(
        provider=DEFAULT_CONFIG["llm_provider"],
        model=DEFAULT_CONFIG["deep_think_llm"],
        base_url=DEFAULT_CONFIG.get("backend_url"),
    )
    raw_llm = llm_client.get_llm()
    structured_llm = bind_structured(raw_llm, AllocationPlan, "Portfolio Constructor (synthesize_only)")

    print(f"[INFO] structured binding: {'OK' if structured_llm else 'FELL BACK TO FREE-TEXT'}")
    print("[INFO] invoking structured synthesis ...")

    t0 = time.time()
    plan: AllocationPlan | None = None
    if structured_llm is not None:
        try:
            plan = structured_llm.invoke(prompt)
            print(f"[INFO] structured OK in {time.time()-t0:.1f}s")
        except Exception as exc:
            print(f"[WARN] structured failed in {time.time()-t0:.1f}s")
            traceback.print_exc()

    if plan is None:
        print("[INFO] retrying free-text path ...")
        t1 = time.time()
        try:
            resp = raw_llm.invoke(prompt)
            raw = getattr(resp, "content", str(resp))
            print(f"[INFO] free-text returned {len(raw)} chars in {time.time()-t1:.1f}s")
            # Save raw response for debugging
            (out_dir / "synthesis_raw.txt").write_text(raw, encoding="utf-8")
            print(f"[WRITE_DONE] {out_dir / 'synthesis_raw.txt'}")

            # Try to parse JSON out of the response
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end <= start:
                print("[ERROR] no {...} JSON block in free-text response")
                return 2
            json_block = raw[start : end + 1]
            try:
                plan = AllocationPlan.model_validate_json(json_block)
                print(f"[INFO] free-text parse OK ({len(json_block)} chars JSON)")
            except Exception as exc:
                print(f"[ERROR] free-text JSON parse failed:")
                traceback.print_exc()
                return 3
        except Exception as exc:
            print(f"[ERROR] free-text invocation itself failed:")
            traceback.print_exc()
            return 4

    plan_md = render_allocation_plan(plan)
    plan_json = json.dumps(plan.model_dump(mode="json"), indent=2)

    (out_dir / "plan.md").write_text(plan_md, encoding="utf-8")
    (out_dir / "plan.json").write_text(plan_json, encoding="utf-8")
    print(f"[WRITE_DONE] {out_dir / 'plan.md'}")
    print(f"[WRITE_DONE] {out_dir / 'plan.json'}")

    print()
    print("===== FINAL ALLOCATION =====")
    for a in plan.allocations:
        print(f"{a.ticker:16s} {a.target_weight:>6.1%}   {a.action}")
    print(f"{'CASH':16s} {plan.cash_weight:>6.1%}   hold")
    print("============================")
    return 0


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else _dt.date.today().isoformat()
    sys.exit(main(date))
