# GX10 Diagnostic Conclusions and Remediation — 2026-05-11

Three real problems surfaced in the May 10–11 GX10 smoke runs. This document records the findings and the remediation in each layer.

## Problem 1 — vLLM 35B servers add ~35s of overhead to every request

**Evidence**:
- Direct curl to `/v1/chat/completions` returns in 1.2-1.5s on the 35B vLLM cells (both 90c5 and gx10-2).
- The same endpoint, called via `langchain_openai.ChatOpenAI` with `with_structured_output`, takes 37-45s.
- Disabling Qwen3 reasoning mode (`enable_thinking: false`) had no effect — latency unchanged. Rules out CoT-token generation as the dominant cost.
- vLLM `/metrics` shows the cells idle when not under request.
- Cold/warm timing ratio is ~1.0, ruling out guided-decoding grammar compilation.

**Root cause**: The vLLM servers were launched with:

```
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  ...
```

Two issues with this configuration for our use case:

1. `--enable-auto-tool-choice` routes every request — including pure `response_format=json_schema` structured-output requests — through the tool-calling pipeline. OptionsAgents uses structured output, not tool calling. The tool-calling middleware adds fixed overhead per request even when no tools are involved.

2. `--tool-call-parser qwen3_coder` is the parser for the *Qwen3 coder* variant. The cells serve the *instruct* variant (`Qwen3.6-35B-A3B-FP8` is instruct). The parser is mismatched to the model, which may cause additional fallback paths or errored-but-recovered handling per request.

**Fix**: Relaunch the vLLM 35B servers without the tool-calling flags:

```
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 ...
```

Drop `--enable-auto-tool-choice` and `--tool-call-parser qwen3_coder` entirely. Preserve everything else.

**Risk and mitigation**: If other clients on the cluster depend on tool-calling against these endpoints, relaunching breaks them. Check usage first. If tool-calling is needed elsewhere, run a second vLLM instance on a different port without the tool-calling flags and point OptionsAgents at that instance.

**Expected outcome**: T3 latency drops from 37-45s to under 3s on both 35B cells. T4a and T4b on the 35Bs may also start passing — earlier "truncation" and "timeout" failures were likely tokens-budget-exhausted-mid-CoT, not real capability gaps.

---

## Problem 2 — llama.cpp 122B returns reasoning in `reasoning_content`, not `content`

**Evidence**:
- Direct curl to the 122B endpoint shows responses with populated `reasoning_content` field and empty `content` field for Qwen3-class prompts.
- LangChain's `ChatOpenAI` client reads `content` (standard OpenAI API field). Gets empty string. Returns empty `AIMessage`.
- `with_structured_output(OptionsProposal).invoke(prompt)` then either raises a parsing error or — depending on langchain version — returns a Pydantic instance with whatever defaults the parsing layer constructs.

**Root cause**: llama.cpp's OpenAI-compat server, when serving Qwen3 reasoning-mode models, emits the model's response into a non-standard `reasoning_content` field rather than the standard `content` field. This is a known divergence between llama.cpp and other OpenAI-compat runtimes (vLLM, TGI, OpenAI proper) for reasoning models.

**Fix options** (in order of preference):

1. **Server-side**: launch llama.cpp with a flag that merges or redirects reasoning into the standard content field. Check `llama-server --help | grep -i reason` for available flags. Recent builds expose options like `--reasoning-format` or `--no-reasoning-stream`.

2. **Server-side alternative**: upgrade llama.cpp to a version where Qwen3 support emits to `content` by default.

3. **Client-side fallback**: write a small `ChatOpenAI` subclass that reads `additional_kwargs["reasoning_content"]` into `content` when `content` is empty. Keeps OptionsAgents portable across runtimes but adds a wart to the client code.

The server-side fix is preferred because it isolates the workaround to one place (the runtime config) rather than spreading runtime-specific logic across our agent code.

---

## Problem 3 — Smoke test could silently accept empty model output as success

**Evidence**:
- Earlier smoke runs against the 122B reported `T4a ✓` while debugging showed the model was actually returning empty content. The "pass" reflected default-filled or structurally-trivial proposals slipping past the test's field-level checks under specific conditions.

**Root cause**: The smoke test called `llm.with_structured_output(OptionsProposal).ainvoke(prompt)` and inspected the returned Pydantic object directly. This loses the raw response — there's no way to detect "the model returned empty content, langchain constructed a placeholder" vs "the model returned a real, useful response."

**Fix (this commit)**:

1. Smoke test now uses `with_structured_output(OptionsProposal, include_raw=True)`. The result is a `{"raw", "parsed", "parsing_error"}` dict, letting us inspect the wire-level response before trusting the parsed object.

2. All field-level assertions are centralized in `tradingagents/agents/wheel_schema_assertions.py`. The module exports:
   - `assert_raw_content_nonempty(raw)` — catches the llama.cpp Qwen3 issue at the right layer
   - `assert_proposal_not_none(proposal)` — guards against `None` returns
   - `assert_rationale_substantive(proposal)` — rejects single-word "ok" rationales
   - `assert_valid_sell_put(proposal)` — full check for T4a-style trade proposals
   - `assert_valid_no_op(proposal, expected_reason)` — full check for T4b-style NO_OP proposals
   - `assert_structured_response_complete(raw_result, expected_action, expected_no_op_reason=None)` — end-to-end assertion used by the smoke test

3. The assertion module itself is unit-tested by `scripts/check_wheel_schemas.py` (now 24 invariants, up from 10). The added negative tests prove each assertion correctly rejects: `None` raw responses, empty content, reasoning-only content (the actual May-2026 llama.cpp Qwen3 failure), `None` proposals, wrong-type proposals, short rationales, null strikes on SELL_PUT, wrong no_op_reason on NO_OP, parsing errors, and the reasoning-only-content end-to-end case.

**Pattern going forward**: any future agent-output test (Stage 3 graph-output tests, Stage 5 shadow-mode regression) imports from `wheel_schema_assertions`. The "what makes an LLM-generated proposal trustworthy" definition lives in one place, versioned in git, with negative tests proving it works.

---

## Expected new smoke results after all three fixes

Run order: fix Problem 1 → fix Problem 2 → re-run smoke (Problem 3 fix is already in this commit).

After only Problem 1 (vLLM relaunch):
- T3 on 90c5 and gx10-2 drops to 1-3s
- T4a on 90c5 likely passes; gx10-2 likely passes
- T4b on 90c5 and gx10-2 may pass or may still need context-following capability we haven't proven

After Problem 1 + Problem 2 (vLLM relaunch + llama.cpp reasoning fix):
- T2 on 122B works reliably (no more transient timeouts from empty `content`)
- T3 on 122B stays ~1s
- T4a on 122B genuinely passes (not via empty-content fallback)
- T4b on 122B genuinely passes

After Problem 1 + Problem 2 + Problem 3 (this commit):
- Any "false positive" pass becomes structurally impossible. A test that prints "✓" on a cell means that cell actually returned non-empty content, langchain parsed it without error, and the parsed object has substantive values matching expectations.

That last property is what makes Stage 3 model routing decisions trustworthy.
