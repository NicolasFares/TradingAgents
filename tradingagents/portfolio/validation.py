"""Pre-flight ticker validation against yfinance.

Catches symbols that yfinance silently 404s on (delisted, wrong suffix,
typo) BEFORE the 60+ minute per-ticker analyst phase runs and the LLM
fills the data void with confabulation.

Two-stage check per ticker (cheapest first):
  1. ``yf.Ticker(sym).fast_info.last_price`` — ~50-100ms when it works
  2. ``yf.Ticker(sym).history(period="5d")`` — ~500-1000ms fallback

Suggestion sources, ranked:
  1. Static suffix permutations (deterministic, no extra HTTP):
     - Dots-to-dashes in suffix infix:  ``BTCX.B.TO`` -> ``BTCX-B.TO``
     - Drop middle segment:            ``BTCX.B.TO`` -> ``BTCX.TO``
     - Common European venues for bare: ``VUAA`` -> ``VUAA.L``, ``.MI``, ``.DE``, ``.AS``, ``.SW``, ``.PA``
  2. ``yfinance.Search`` — broad fuzzy match

Each candidate is itself validated before being shown to the user, so we
never suggest a symbol that's also broken.

Results cache to ``~/.tradingagents/cache/ticker_validation.json`` with a
24-hour TTL.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal, Optional

import yfinance as yf


logger = logging.getLogger(__name__)


CACHE_PATH = Path("~/.tradingagents/cache/ticker_validation.json").expanduser()
CACHE_TTL = timedelta(hours=24)

# Common European exchanges we try for bare (no-suffix) symbols that look
# like UCITS ETFs (4-5 char alphabetic). Ordered by relative prevalence.
_EUROPE_SUFFIXES = (".L", ".MI", ".DE", ".AS", ".SW", ".PA")

# Delay between consecutive yfinance HTTP calls to stay polite with their
# unofficial scraping endpoint. ~7 tickers * up to 5 candidate validations
# ~= 35 calls. 0.15s spacing keeps total wall time well under 10s.
_REQUEST_SPACING_S = 0.15


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TickerCandidate:
    symbol: str
    long_name: str = ""
    exchange: str = ""
    quote_type: str = ""
    last_price: Optional[float] = None
    avg_volume: Optional[float] = None


@dataclass
class TickerValidation:
    original: str
    status: Literal["ok", "needs_substitution", "unresolved"]
    resolved: Optional[TickerCandidate] = None
    suggestions: list[TickerCandidate] = field(default_factory=list)
    diagnostic: str = ""


class BasketValidationError(Exception):
    """Raised when one or more tickers cannot be resolved and the preflight
    policy is to abort (interactive cancel, or non-interactive abort mode)."""

    def __init__(self, message: str, unresolved: list[str]):
        super().__init__(message)
        self.unresolved = unresolved


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("ticker validation cache unreadable (%s); starting fresh", exc)
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _cache_key(symbol: str) -> str:
    return symbol.upper().strip()


def _is_fresh(entry: dict) -> bool:
    ts = entry.get("_ts")
    if not ts:
        return False
    try:
        age = datetime.now() - datetime.fromisoformat(ts)
    except ValueError:
        return False
    return age < CACHE_TTL


def _validation_to_dict(v: TickerValidation) -> dict:
    return {
        "original": v.original,
        "status": v.status,
        "resolved": dataclasses.asdict(v.resolved) if v.resolved else None,
        "suggestions": [dataclasses.asdict(s) for s in v.suggestions],
        "diagnostic": v.diagnostic,
        "_ts": datetime.now().isoformat(),
    }


def _validation_from_dict(d: dict) -> TickerValidation:
    return TickerValidation(
        original=d["original"],
        status=d["status"],
        resolved=TickerCandidate(**d["resolved"]) if d.get("resolved") else None,
        suggestions=[TickerCandidate(**s) for s in d.get("suggestions", [])],
        diagnostic=d.get("diagnostic", ""),
    )


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


def _is_real_number(x) -> bool:
    if x is None:
        return False
    try:
        return not math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def _probe_via_fast_info(symbol: str) -> Optional[TickerCandidate]:
    """Stage 1: yfinance fast_info. Returns None if the ticker is unknown."""
    try:
        ticker = yf.Ticker(symbol)
        fi = ticker.fast_info
        last_price = getattr(fi, "last_price", None)
        if not _is_real_number(last_price):
            return None
        return TickerCandidate(
            symbol=symbol,
            exchange=getattr(fi, "exchange", "") or "",
            quote_type=getattr(fi, "quote_type", "") or "",
            last_price=float(last_price),
        )
    except Exception:
        return None


def _probe_via_history(symbol: str) -> Optional[TickerCandidate]:
    """Stage 2: 5-day history fallback. More reliable but slower."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        last_price = float(hist["Close"].iloc[-1])
        avg_vol = None
        if "Volume" in hist.columns:
            try:
                avg_vol = float(hist["Volume"].mean())
            except Exception:
                pass
        return TickerCandidate(symbol=symbol, last_price=last_price, avg_volume=avg_vol)
    except Exception:
        return None


def _enrich(candidate: TickerCandidate) -> TickerCandidate:
    """Best-effort: pull longName / exchange / quoteType from ticker.info.

    yfinance's .info is noisy and slow; we only call it after we've already
    confirmed the symbol resolves, so the noise cost is bounded.
    """
    try:
        info = yf.Ticker(candidate.symbol).info or {}
    except Exception:
        info = {}
    if not candidate.long_name:
        candidate.long_name = info.get("longName") or info.get("shortName") or ""
    if not candidate.exchange:
        candidate.exchange = info.get("exchange") or ""
    if not candidate.quote_type:
        candidate.quote_type = info.get("quoteType") or ""
    return candidate


def _probe_yfinance(symbol: str) -> Optional[TickerCandidate]:
    """Two-stage probe with brief politeness sleep."""
    candidate = _probe_via_fast_info(symbol)
    time.sleep(_REQUEST_SPACING_S)
    if candidate is None:
        candidate = _probe_via_history(symbol)
        time.sleep(_REQUEST_SPACING_S)
    if candidate is None:
        return None
    return _enrich(candidate)


# ---------------------------------------------------------------------------
# Suggestion generation
# ---------------------------------------------------------------------------


def _suffix_permutations(symbol: str) -> list[str]:
    """Deterministic alternative symbols to try before falling back to search."""
    out: list[str] = []

    if "." in symbol:
        parts = symbol.split(".")
        # Dots-to-dashes in the infix: BTCX.B.TO -> BTCX-B.TO
        if len(parts) >= 3:
            joined = "-".join(parts[:-1]) + "." + parts[-1]
            out.append(joined)
            # Drop the middle segment: BTCX.B.TO -> BTCX.TO
            out.append(parts[0] + "." + parts[-1])
        # Just the base: BTCX.B.TO -> BTCX
        if parts[0] not in out:
            out.append(parts[0])
    else:
        # Bare symbol — try common European venues.
        for suffix in _EUROPE_SUFFIXES:
            out.append(symbol + suffix)

    # Deduplicate while preserving order.
    seen = set()
    result = []
    for s in out:
        key = s.upper()
        if key in seen or key == symbol.upper():
            continue
        seen.add(key)
        result.append(s)
    return result


def _search_suggestions(query: str, limit: int = 8) -> list[str]:
    try:
        hits = yf.Search(query, max_results=limit).quotes or []
    except Exception as exc:
        logger.warning("yfinance Search failed for %s: %s", query, exc)
        return []
    return [h.get("symbol") for h in hits if h.get("symbol")]


# ---------------------------------------------------------------------------
# Public: per-ticker validate
# ---------------------------------------------------------------------------


def validate_ticker(
    symbol: str,
    *,
    asset_type: str = "stock",
    use_cache: bool = True,
    cache: Optional[dict] = None,
    max_suggestions: int = 3,
) -> TickerValidation:
    """Validate one ticker; return resolution + ranked suggestions on failure."""
    cache = cache if cache is not None else (_load_cache() if use_cache else {})
    key = _cache_key(symbol)

    if use_cache and key in cache and _is_fresh(cache[key]):
        return _validation_from_dict(cache[key])

    # Stage 1: try the symbol as-given.
    candidate = _probe_yfinance(symbol)
    if candidate is not None:
        result = TickerValidation(
            original=symbol,
            status="ok",
            resolved=candidate,
            diagnostic=f"resolved as-given ({candidate.long_name or candidate.symbol})",
        )
        if use_cache:
            cache[key] = _validation_to_dict(result)
            _save_cache(cache)
        return result

    # Stage 2: generate + validate suggestions.
    suggestions: list[TickerCandidate] = []
    tried = {symbol.upper()}

    for candidate_sym in _suffix_permutations(symbol):
        if candidate_sym.upper() in tried:
            continue
        tried.add(candidate_sym.upper())
        cand = _probe_yfinance(candidate_sym)
        if cand is not None:
            suggestions.append(cand)
        if len(suggestions) >= max_suggestions:
            break

    # Fall back to yfinance Search if we don't have enough.
    if len(suggestions) < max_suggestions:
        for sym in _search_suggestions(symbol):
            if sym.upper() in tried:
                continue
            tried.add(sym.upper())
            cand = _probe_yfinance(sym)
            if cand is not None:
                suggestions.append(cand)
            if len(suggestions) >= max_suggestions:
                break

    result = TickerValidation(
        original=symbol,
        status="needs_substitution" if suggestions else "unresolved",
        suggestions=suggestions,
        diagnostic=(
            f"{symbol} not found on yfinance; "
            + (f"{len(suggestions)} suggestion(s) available" if suggestions else "no alternatives found")
        ),
    )
    if use_cache:
        cache[key] = _validation_to_dict(result)
        _save_cache(cache)
    return result


# ---------------------------------------------------------------------------
# Public: basket preflight
# ---------------------------------------------------------------------------


def preflight_basket(
    tickers: list[str],
    *,
    interactive: bool = True,
    on_unresolved: Literal["abort", "skip", "auto"] = "abort",
    use_cache: bool = True,
) -> list[str]:
    """Validate every ticker and return the final basket to run.

    interactive=True:  prompt the user (via questionary) for each unresolved
                       ticker; "abort" choice raises BasketValidationError.
    interactive=False: apply ``on_unresolved`` mechanically:
                       - "abort": raise BasketValidationError listing all failures
                       - "skip":  drop unresolved tickers, log a warning per drop
                       - "auto":  silently take the top suggestion when one exists,
                                  abort when there is none
    """
    cache = _load_cache() if use_cache else {}

    print(f"[PREFLIGHT] Validating {len(tickers)} tickers against yfinance...", flush=True)
    results: dict[str, TickerValidation] = {}
    for t in tickers:
        v = validate_ticker(t, use_cache=use_cache, cache=cache)
        results[t] = v
        print(_format_status_line(t, v), flush=True)

    if use_cache:
        _save_cache(cache)

    final: list[str] = []
    abort_list: list[str] = []

    for t, v in results.items():
        if v.status == "ok":
            final.append(t)
            continue

        if interactive:
            chosen = _interactive_resolve(t, v, cache=cache, use_cache=use_cache)
            if chosen is not None:
                final.append(chosen)
        else:
            if on_unresolved == "abort":
                abort_list.append(t)
            elif on_unresolved == "skip":
                logger.warning("[PREFLIGHT] skipping unresolved ticker %s", t)
            elif on_unresolved == "auto":
                if v.suggestions:
                    chosen = v.suggestions[0].symbol
                    logger.info("[PREFLIGHT] auto-substituting %s -> %s", t, chosen)
                    final.append(chosen)
                else:
                    abort_list.append(t)

    if abort_list:
        raise BasketValidationError(
            f"Unresolved tickers: {abort_list}", unresolved=abort_list
        )

    print(
        f"[PREFLIGHT] Final basket ({len(final)} tickers): "
        + (", ".join(final) if final else "(empty)"),
        flush=True,
    )

    if interactive and final:
        if not _confirm("Proceed with this basket?"):
            print("[PREFLIGHT] Aborted by user.", flush=True)
            return []

    return final


# ---------------------------------------------------------------------------
# Display + prompt helpers
# ---------------------------------------------------------------------------


def _format_status_line(original: str, v: TickerValidation) -> str:
    if v.status == "ok" and v.resolved is not None:
        name = v.resolved.long_name or "(name unavailable)"
        ex = f" [{v.resolved.exchange}]" if v.resolved.exchange else ""
        return f"  {original:14s} OK         {name}{ex}"
    if v.status == "needs_substitution":
        sugs = ", ".join(s.symbol for s in v.suggestions)
        return f"  {original:14s} NOT FOUND  suggestions: {sugs}"
    return f"  {original:14s} NOT FOUND  no suggestions available"


def _interactive_resolve(
    original: str,
    v: TickerValidation,
    *,
    cache: dict,
    use_cache: bool,
) -> Optional[str]:
    """Prompt the user; return the chosen replacement symbol or None to skip.

    Raises BasketValidationError if the user picks 'abort'.
    """
    import questionary

    while True:
        choices = []
        for s in v.suggestions:
            label_extras = " — ".join(
                filter(None, [s.long_name, f"[{s.exchange}]" if s.exchange else ""])
            )
            label = f"{s.symbol}" + (f"  {label_extras}" if label_extras else "")
            choices.append(questionary.Choice(title=label, value=s.symbol))
        choices.append(questionary.Choice(title="Enter a different ticker manually", value="__manual__"))
        choices.append(questionary.Choice(title="Skip this ticker", value="__skip__"))
        choices.append(questionary.Choice(title="Abort the whole basket run", value="__abort__"))

        answer = questionary.select(
            f"{original} — pick a replacement:",
            choices=choices,
        ).ask()

        if answer is None or answer == "__abort__":
            raise BasketValidationError(
                f"User aborted at {original}", unresolved=[original]
            )
        if answer == "__skip__":
            return None
        if answer == "__manual__":
            manual = questionary.text(f"Replacement symbol for {original}:").ask()
            if not manual:
                continue
            manual = manual.strip().upper()
            print(f"  validating {manual} ...", flush=True)
            v2 = validate_ticker(manual, use_cache=use_cache, cache=cache)
            if v2.status == "ok":
                print(_format_status_line(manual, v2), flush=True)
                if use_cache:
                    _save_cache(cache)
                return manual
            print(_format_status_line(manual, v2), flush=True)
            # loop again, re-show original suggestions
            continue

        return answer  # one of the suggested symbols


def _confirm(message: str) -> bool:
    import questionary

    answer = questionary.confirm(message, default=True).ask()
    return bool(answer)
