# Smoke Test Update — Wire-Format Value Redesign

**Date**: 2026-05-10
**Trigger**: GX10 smoke test results showed 0/4 models passing T4a `OptionsProposal` trade case. All four returned `action=STO_CALL` with `right=P`, indicating they correctly identified "put" but defaulted to the wrong action enum. Root cause: enum *values* were trader jargon (`STO_PUT`, `BTC_CALL`, `NO_OP`) which gave the models no semantic anchor to map prompt vocabulary onto.

## What changed in `tradingagents/agents/wheel_schemas.py`

Python enum *names* are unchanged — no downstream code needs to update. Only the *string values* (what the LLM sees as allowed schema values) changed:

| Enum                 | Old value                  | New value                                          |
|----------------------|----------------------------|----------------------------------------------------|
| `WheelAction.SELL_PUT`        | `STO_PUT`                  | `sell_cash_secured_put`                            |
| `WheelAction.BUY_PUT`         | `BTC_PUT`                  | `buy_put_to_close`                                 |
| `WheelAction.SELL_CALL`       | `STO_CALL`                 | `sell_covered_call`                                |
| `WheelAction.BUY_CALL`        | `BTC_CALL`                 | `buy_call_to_close`                                |
| `WheelAction.ROLL_PUT`        | `ROLL_PUT`                 | `roll_put_to_later_expiry`                         |
| `WheelAction.ROLL_CALL`       | `ROLL_CALL`                | `roll_call_to_later_expiry`                        |
| `WheelAction.NO_OP`           | `NO_OP`                    | `do_not_trade`                                     |
| `OptionRight.PUT`             | `P`                        | `put`                                              |
| `OptionRight.CALL`            | `C`                        | `call`                                             |
| `CycleStage.CASH`             | `CASH`                     | `cash_no_position`                                 |
| `CycleStage.SHORT_PUT`        | `SHORT_PUT`                | `short_put_open`                                   |
| `CycleStage.ASSIGNED`         | `ASSIGNED`                 | `shares_assigned`                                  |
| `CycleStage.SHORT_CALL`       | `SHORT_CALL`               | `covered_call_open`                                |
| `NoOpReason.EARNINGS_BLACKOUT`| `EARNINGS_BLACKOUT`        | `earnings_blackout_window`                         |
| `NoOpReason.CHAIN_UNAVAILABLE`| `CHAIN_UNAVAILABLE`        | `option_chain_unavailable`                         |
| (full list in `wheel_schemas.py`)                                                                  |

The `OptionsProposal.action` and `OptionsProposal.no_op_reason` `Field(description=...)` annotations were also expanded with explicit prompt-to-value mapping instructions. These descriptions become part of the JSON schema the model sees, so they function as inline guidance the model reads during structured-output generation.

## What `scripts/smoke_gx10.py` needs to update

**No changes to the test logic, prompts, or YAML config are required.** The whole point of the redesign is that the prompts can stay in natural English and the model maps them to the new self-describing values without help.

The only updates required to `smoke_gx10.py`:

1. **T4a assertion**: change any reference to `WheelAction.SELL_PUT.value == "STO_PUT"` to the comparison-by-enum-member form, e.g.:

   ```python
   # Before (brittle — checks the string):
   assert result["action"] == "STO_PUT"

   # After (model-agnostic — checks the enum member):
   assert proposal.action == WheelAction.SELL_PUT
   ```

   Comparing parsed `OptionsProposal` instances by enum member instead of string value insulates the test from any future wire-format changes.

2. **T4b assertion**: same pattern for `NO_OP`:

   ```python
   assert proposal.action == WheelAction.NO_OP
   assert proposal.no_op_reason == NoOpReason.EARNINGS_BLACKOUT
   ```

3. **Add a cold-vs-warm T3 timing pair** (optional but recommended given the 37s T3 latency on 90c5):

   ```python
   # Run T3 twice in a row — first call pays guided-decoding grammar compile cost,
   # second call hits the compiled-grammar cache. Production morning scans hit
   # the same schema 10-20 times, so the warm number is the production-relevant one.
   t3_cold_ms, result_cold = time_call(lambda: llm.with_structured_output(HealthCheck).invoke(...))
   t3_warm_ms, result_warm = time_call(lambda: llm.with_structured_output(HealthCheck).invoke(...))
   record["T3_cold_ms"] = t3_cold_ms
   record["T3_warm_ms"] = t3_warm_ms
   ```

   Report both in the JSON log. If `t3_warm_ms < t3_cold_ms / 3`, the grammar-compile hypothesis is confirmed and we know production latency will be fine.

## Expected new results

Re-running the smoke test against the redesigned schemas should yield:

- **T4a (trade case)**: 122B passes, 90c5 likely passes, 35B and gemma4 *probably* pass. The redesigned `action` field has explicit prompt-to-value mapping in its description, so even smaller models have an instruction to follow.
- **T4b (NO_OP case)**: 122B still passes, 90c5 likely passes. Smaller models may still fall back to `sell_cash_secured_put` because they have weaker context-following. That's an acceptable Stage 3 outcome — route the analyst role to the larger models and use the smaller ones for the utility / scanner-interface roles.

If T4a still fails universally after the redesign, the problem is *not* enum values and we need to look at vLLM's guided decoding configuration or the prompts themselves. But the diagnostic data strongly suggests the redesign will move the needle.

## Why this matters beyond the smoke test

The same self-describing-value principle applies to Stage 3 when we wire the agent graph. Any new enum we add (e.g., a `ScanCandidateSource` enum distinguishing "scanner_proposed" vs "user_specified" vs "memory_recommended") should follow the plain-English-value convention from the start. Internal Python names stay terse for code readability; wire-format values stay self-describing for model legibility. This decouples the system from any specific model's quirks and is what makes "swap a model later" actually work.
