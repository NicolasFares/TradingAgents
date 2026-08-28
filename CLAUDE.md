# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TradingAgents is a LangGraph-orchestrated multi-agent trading framework. A single `propagate(ticker, date)` call walks: parallel **analysts** (market / sentiment / news / fundamentals) → sequential **bull/bear researcher debate** → **Research Manager** synthesis → **Trader** proposal → sequential **risk debate** (aggressive / neutral / conservative) → **Portfolio Manager** approval — producing a structured `PortfolioDecision`. v0.2.5 adds structured-output Pydantic schemas, LangGraph checkpoint resume, multi-provider LLM support (OpenAI / Anthropic / Google / xAI / DeepSeek / Qwen / GLM / MiniMax / Azure / Ollama / OpenRouter), persistent decision memory log, and ticker path-traversal hardening.

## Common commands

- Install: `pip install .` (Python ≥3.10; conda env with 3.13 is the README's recommendation).
- All tests: `pytest tests/`
- Single test: `pytest tests/test_<name>.py::<TestClass>::<test_method>`
- Programmatic run: `python main.py` (uses `DEFAULT_CONFIG` + `TradingAgentsGraph.propagate(ticker, date)`)
- Interactive CLI: `tradingagents` (Typer entry point) or `python -m cli.main`

## Architecture (the parts you can't infer from one file)

- `tradingagents/graph/trading_graph.py` — `TradingAgentsGraph` orchestrator. Constructs deep- and quick-thinking LLM clients via the factory, exposes `propagate()`. Read this first.
- `tradingagents/graph/setup.py` — `GraphSetup` wires every agent node into the LangGraph `StateGraph`. The pipeline shape lives here.
- `tradingagents/graph/analyst_execution.py` — toggles between parallel and sequential analyst execution (`analyst_concurrency_limit`).
- `tradingagents/graph/checkpointer.py` — SQLite-backed LangGraph checkpoint resume. Enable per-run via `TRADINGAGENTS_CHECKPOINT_ENABLED=true`.
- `tradingagents/graph/conditional_logic.py` / `propagation.py` / `reflection.py` — node routing, debate-round termination, post-run reflection + memory append.
- `tradingagents/agents/` — agent factories grouped by team: `analysts/`, `researchers/`, `managers/` (research + portfolio), `trader/`, `risk_mgmt/`.
- `tradingagents/agents/schemas.py` — `ResearchPlan`, `TraderProposal`, `PortfolioDecision` Pydantic models. These drive structured output and are auto-rendered back to markdown for downstream prompts.
- `tradingagents/llm_clients/factory.py` — `create_llm_client(provider, model, base_url, **kwargs)` dispatcher. OpenAI-compatible providers (`openai`, `xai`, `deepseek`, `qwen`, `qwen-cn`, `glm`, `glm-cn`, `minimax`, `minimax-cn`, `ollama`, `openrouter`) all share `openai_client.py`; `anthropic_client.py`, `google_client.py`, `azure_client.py` are native.
- `tradingagents/llm_clients/capabilities.py` — chooses the structured-output mode per provider (json_schema vs response_schema vs tool-use) and applies reasoning/thinking knobs.
- `tradingagents/llm_clients/model_catalog.py` — curated model lists per provider (CLI dropdown). `ollama` and `openrouter` skip validation (`validators.py`) and accept any model string.
- `tradingagents/dataflows/interface.py` — data-vendor abstraction. Defaults to `yfinance`; per-category and per-tool overrides via `data_vendors` / `tool_vendors` in config (e.g. swap fundamentals to `alpha_vantage`).
- `cli/main.py` (Typer) + `cli/utils.py` — interactive workflow. When `ollama` is picked, `confirm_ollama_endpoint()` surfaces the resolved `OLLAMA_BASE_URL`.

## Configuration model

- `tradingagents/default_config.py` is the single source of truth.
- Environment overrides via `TRADINGAGENTS_*` — mapped in `_ENV_OVERRIDES` (lines 10–20): `LLM_PROVIDER`, `DEEP_THINK_LLM`, `QUICK_THINK_LLM`, `LLM_BACKEND_URL`, `OUTPUT_LANGUAGE`, `MAX_DEBATE_ROUNDS`, `MAX_RISK_ROUNDS`, `CHECKPOINT_ENABLED`, `BENCHMARK_TICKER`. Values are coerced to the existing default's type (so `"true"`/`"3"` work).
- Provider API keys via `.env` (see `.env.example`). `OLLAMA_BASE_URL` overrides the Ollama endpoint at call time (resolved in `openai_client.py:174`, not at import time — so a remote ollama-serve works without code edits).
- Provider thinking knobs: `google_thinking_level`, `openai_reasoning_effort`, `anthropic_effort` (config keys, not env vars).
- User state lives under `~/.tradingagents/` — `logs/`, `cache/`, `memory/trading_memory.md` (the persistent decision log injected into future prompts).
