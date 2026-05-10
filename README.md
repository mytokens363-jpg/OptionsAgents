# OptionsAgents

**Multi-agent LLM framework for wheel-strategy options trading.**

A fork of [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents), adapted for cash-secured puts → assignment → covered calls cycle management against Interactive Brokers via IB Gateway. Designed to integrate with the existing **WheelBot** execution pipeline as a parallel proposal-and-reflection layer, not a replacement.

> **Upstream fork point**: TradingAgents commit `afdc6d4ec1008da88a8004e0d76a34381daab9ef`
> See the original [`README.md`](./README.upstream.md) for upstream-specific docs.

---

## Why fork TradingAgents

TradingAgents already encodes four pieces of architecture that map cleanly to the wheel strategy:

| Upstream feature                                            | OptionsAgents role                                              |
|-------------------------------------------------------------|-----------------------------------------------------------------|
| LangGraph orchestration with typed Pydantic schemas         | `OptionsProposal` / `WheelDecision` as new structured artifacts |
| Tool node pattern (`get_stock_data`, `get_indicators`, …)   | Drop-in replacement target for IBKR/state-file tools            |
| Append-only memory log with pending→resolved reflection     | Extends to wheel-cycle outcomes (`CycleOutcome`)                |
| Provider abstraction via OpenAI-compatible `base_url`       | Plugs directly into the GX10 inference cluster                  |
| `backtrader>=1.9.78.123` already in deps                    | Available for OptionsAgents replay validation                   |

What changes: the **decision artifacts** (replaced with wheel semantics), the **tool layer** (IBKR + state files instead of vendor stock data), and the **memory keying** (setup-signature buckets vs returns-vs-SPY benchmarking).

What stays: orchestration, structured-output binding, provider abstraction, the LangGraph state-machine + checkpointing, the markdown report format.

---

## Staged build plan

| Stage   | Scope                                                                                                      | Status     |
|---------|------------------------------------------------------------------------------------------------------------|------------|
| **0**   | Fork prep: rename project, reset git history, add `ib_insync` dep, verify GX10 endpoint                    | ✅ Done    |
| **1**   | Wheel schemas: `OptionsProposal`, `WheelState`, `WheelDecision`, `CycleOutcome` (+ structured failure)     | ✅ Done    |
| **2**   | IBKR + state-file tools: `get_account_state`, `get_option_chain`, `get_state_files`, `get_earnings`        | Not started |
| **3**   | Agent graph: Scanner / Analyst / Risk / Portfolio / Reflection roles + LangGraph edges                      | Not started |
| **4**   | Memory extension: wheel-cycle outcome reflection keyed by setup signature                                   | Not started |
| **5**   | Shadow mode: run alongside existing WheelBot scanner, log proposals to `proposals.jsonl`, no execution      | Not started |
| **6**   | Backtrader validation: replay agent decisions against historical option chain data                          | Not started |
| **7**   | Promotion path: config-flag cutover, legacy scanner remains runnable, full rationale chain via Telegram     | Not started |

**Invariant carried through every stage**: the existing WheelBot scanner/executor stays the source of truth until OptionsAgents earns promotion via Stages 5–6.

---

## Stage 1 deliverable: wheel schemas

`tradingagents/agents/wheel_schemas.py` defines:

- `OptionsProposal` — per-symbol proposal from an analyst, including a structured `NO_OP` path
- `WheelState` — read-only account snapshot built from IBKR + state files
- `WheelDecision` — final Portfolio Manager output with risk-gate audit trail
- `CycleOutcome` — realized leg outcome for the reflection/memory loop

### Anti-silent-failure invariant

The most important design choice in Stage 1: **a proposal cannot validate as "do nothing" without a structured reason**. This is the schema-layer answer to the April 2026 scanner regression, where `scan_ticker()` silently exited on empty `reqTickers` returns and masked a 3-week failure of new put openings.

```python
# Validates:
OptionsProposal(symbol="TSLA", cycle_stage=CycleStage.CASH,
                action=WheelAction.NO_OP,
                no_op_reason=NoOpReason.EARNINGS_BLACKOUT,
                rationale="Earnings 2026-05-12, within 7d window.")

# Raises ValidationError — caught at graph runtime, not in production at 09:35:
OptionsProposal(symbol="TSLA", cycle_stage=CycleStage.CASH,
                action=WheelAction.NO_OP,
                rationale="meh, skipping")
```

Run `python3 scripts/check_wheel_schemas.py` to exercise all 10 schema invariants.

---

## Local setup (mirrors WheelBot conventions)

```bash
cd OptionsAgents
python3 -m venv .venv
source .venv/bin/activate
pip install -e .          # pyproject.toml dependencies, including ib_insync
```

GX10 endpoint smoke test (Stage 0 acceptance):

```bash
export TRADINGAGENTS_BACKEND_URL="http://<gx10-host>:<port>/v1"
export OPENAI_API_KEY="<gx10-token>"
python3 scripts/smoke_structured_output.py
```

---

## Relationship to WheelBot

OptionsAgents will live alongside (not inside) the WheelBot repo at `/Users/dariusvitkus/wheelbot/optionsagents/`. Coordination notes:

- **Read-only on state files** through Stage 5. OptionsAgents tools read `trades.jsonl`, `rolls.jsonl`, `regime.json`, `earnings_dates.json` but never write to them.
- **No IBKR write operations** until Stage 7. Stages 2–6 use the IBKR API for state inspection and option chain reads only.
- **Distinct `clientId`** when connecting to IB Gateway, namespaced away from the existing 17+ observed IDs to keep diagnostics clean.
- **Independent Telegram channel** for OptionsAgents proposals during shadow mode, so they're never confused with WheelBot's live alerts.

---

## License

Inherits Apache 2.0 from upstream. See [`LICENSE`](./LICENSE).
