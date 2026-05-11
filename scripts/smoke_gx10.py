#!/usr/bin/env python3
"""GX10 endpoint smoke tests.

Reads gx10_endpoints.yaml, runs 4 tests per endpoint in parallel across
endpoints, serially within each endpoint.

Tests (per cell, run in order):
  T1  Reachability    GET {base_url}/models, 5s timeout, HTTP 200 + data[].id
  T2  Plain chat      langchain_openai.ChatOpenAI, "Reply with exactly the word: pong"
  T3  Structured out  Pydantic HealthCheck via with_structured_output (cold+warm pair)
  T4  OptionsProposal wheel_schemas round-trip
      T4a  Trade case  SELL_PUT
      T4b  NO_OP case   earnings blackout
"""

import asyncio
import sys
import os
import time
import json
import importlib.util
import yaml
import traceback
from datetime import datetime
import aiohttp
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Load wheel_schemas via importlib (avoid agents/__init__ which imports langchain)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCHEMA_PATH = os.path.join(_HERE, "..", "tradingagents", "agents", "wheel_schemas.py")
_spec = importlib.util.spec_from_file_location("wheel_schemas", _SCHEMA_PATH)
_schema_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_schema_mod)

OptionsProposal = _schema_mod.OptionsProposal
WheelAction = _schema_mod.WheelAction
CycleStage = _schema_mod.CycleStage
OptionRight = _schema_mod.OptionRight
NoOpReason = _schema_mod.NoOpReason

# Load wheel_schema_assertions the same way. It imports from
# tradingagents.agents.wheel_schemas, so inject our pre-loaded schemas module
# under that name to avoid the package init pulling in langchain.
_ASSERTION_PATH = os.path.join(_HERE, "..", "tradingagents", "agents", "wheel_schema_assertions.py")
_aspec = importlib.util.spec_from_file_location("wheel_schema_assertions", _ASSERTION_PATH)
_assertions = importlib.util.module_from_spec(_aspec)
sys.modules.setdefault("tradingagents", type(sys)("tradingagents"))
sys.modules.setdefault("tradingagents.agents", type(sys)("tradingagents.agents"))
sys.modules["tradingagents.agents.wheel_schemas"] = _schema_mod
_aspec.loader.exec_module(_assertions)

assert_structured_response_complete = _assertions.assert_structured_response_complete
assert_raw_content_nonempty = _assertions.assert_raw_content_nonempty

# ---------------------------------------------------------------------------
# Pydantic schemas for test 3
# ---------------------------------------------------------------------------

class HealthCheck(BaseModel):
    status: str = Field(description="Either 'ok' or 'degraded'")
    latency_ms: int = Field(description="Round-trip latency in milliseconds")


# ---------------------------------------------------------------------------
# T1 — Reachability
# ---------------------------------------------------------------------------

async def test_reachability(ep: dict, timeout: int) -> dict:
    base_url = ep["base_url"].rstrip("/")
    model_name = ep.get("model", "")
    result = {"test": "T1", "status": "—", "latency_ms": 0, "notes": ""}
    start = time.monotonic()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{base_url}/models") as resp:
                elapsed = time.monotonic() - start
                result["latency_ms"] = round(elapsed * 1000)

                if resp.status == 200:
                    data = await resp.json()
                    ids = [m.get("id") for m in data.get("data", [])]
                    if ids:
                        result["status"] = "✓"
                        result["notes"] = f"models: {', '.join(ids)}"
                        # Check advertised vs expected model name
                        if model_name and model_name not in ids:
                            result["notes"] += f"  ⚠ config says '{model_name}', server says [{', '.join(ids)}]"
                    else:
                        result["status"] = "✗"
                        result["notes"] = "HTTP 200 but no data[] in response"
                elif resp.status == 404:
                    # Fallback: minimal /chat/completions ping
                    try:
                        async with session.post(
                            f"{base_url}/chat/completions",
                            json={
                                "model": model_name,
                                "messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 5,
                            },
                        ) as resp2:
                            if resp2.status == 200:
                                result["status"] = "✓"
                                result["notes"] = f"fallback /chat/completions ping OK"
                            else:
                                result["status"] = "✗"
                                result["notes"] = f"/models 404, fallback returned {resp2.status}"
                    except Exception:
                        result["status"] = "✗"
                        result["notes"] = f"/models 404, fallback also failed"
                else:
                    result["status"] = "✗"
                    result["notes"] = f"HTTP {resp.status}"
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        result["latency_ms"] = round(elapsed * 1000)
        result["status"] = "✗"
        result["notes"] = "timeout"
    except Exception as e:
        result["status"] = "✗"
        result["notes"] = str(e)[:120]
    return result


# ---------------------------------------------------------------------------
# T2 — Plain chat completion
# ---------------------------------------------------------------------------

async def test_plain_chat(ep: dict, timeout: int) -> dict:
    base_url = ep["base_url"].rstrip("/")
    result = {"test": "T2", "status": "—", "latency_ms": 0, "notes": ""}
    try:
        llm = ChatOpenAI(
            model=ep["model"],
            api_key=ep["api_key"],
            base_url=base_url,
            temperature=0,
            max_tokens=200,  # reasoning models burn tokens on chain-of-thought
        )
        start = time.monotonic()
        wrapped = asyncio.wait_for(llm.ainvoke("Reply with exactly the word: pong"), timeout=timeout)
        resp = await wrapped
        elapsed = time.monotonic() - start
        result["latency_ms"] = round(elapsed * 1000)
        content = str(resp.content).strip() if hasattr(resp, "content") else ""
        # Some reasoning models (vLLM Qwen3.6) put CoT in reasoning field,
        # leaving content empty when max_tokens is small. Check reasoning too.
        reasoning = ""
        if hasattr(resp, "additional_kwargs"):
            reasoning = resp.additional_kwargs.get("reasoning", "") or ""
        combined = (content + " " + reasoning).strip()
        if "pong" in combined.lower():
            result["status"] = "✓"
        else:
            result["status"] = "✗"
            result["notes"] = f"expected 'pong', got content={content[:80]} reasoning={reasoning[:80]}"
    except asyncio.TimeoutError:
        result["status"] = "✗"
        result["notes"] = "timeout"
    except Exception as e:
        result["status"] = "✗"
        result["notes"] = str(e)[:120]
    return result


# ---------------------------------------------------------------------------
# T3 — Generic structured output (cold + warm pair)
# ---------------------------------------------------------------------------

async def test_structured_output(ep: dict, timeout: int) -> dict:
    base_url = ep["base_url"].rstrip("/")
    result = {"test": "T3", "status": "—", "latency_ms": 0, "notes": ""}
    try:
        # Disable Qwen3 reasoning on vLLM cells (hypothesis: CoT causes 37s latency)
        llm_kwargs = {
            "model": ep["model"],
            "api_key": ep["api_key"],
            "base_url": base_url,
            "temperature": 0,
            "max_tokens": 2048,
        }
        if ep.get("runtime") == "vllm" and ep.get("model", "").startswith("Qwen3"):
            llm_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        
        llm = ChatOpenAI(**llm_kwargs)
        llm_structured = llm.with_structured_output(HealthCheck)
    except Exception:
        result["status"] = "⊘"
        result["notes"] = "binding_unsupported"
        return result

    prompt = "Report a health check: status ok, latency 42 ms."
    
    # Run twice: first call pays grammar compile cost (vLLM guided decoding),
    # second call hits cached grammar (production-relevant).
    t3_cold_ms, result_cold = 0, None
    t3_warm_ms, result_warm = 0, None
    
    for i, target in enumerate(["cold", "warm"]):
        try:
            t_request_sent = time.monotonic()
            wrapped = asyncio.wait_for(llm_structured.ainvoke(prompt), timeout=timeout)
            obj = await wrapped
            t_response_complete = time.monotonic()
            elapsed = t_response_complete - t_request_sent
            if target == "cold":
                t3_cold_ms = round(elapsed * 1000)
                result_cold = obj
            else:
                t3_warm_ms = round(elapsed * 1000)
                result_warm = obj
        except asyncio.TimeoutError:
            if target == "cold":
                t3_cold_ms = 0
            else:
                t3_warm_ms = 0
        except Exception as e:
            if target == "cold":
                t3_cold_ms = 0
                result["notes"] = f"cold: {str(e)[:80]}"
            else:
                t3_warm_ms = 0
                if result["notes"]:
                    result["notes"] += f", warm: {str(e)[:80]}"
    
    result["status"] = "✓" if (result_cold and result_warm and result_cold.status == "ok" and result_warm.status == "ok") else "✗"
    result["latency_ms"] = t3_warm_ms  # production-relevant
    if result_cold:
        result["T3_cold_ms"] = t3_cold_ms
        result["notes"] += f"; cold={t3_cold_ms}ms" if result["notes"] else f"cold={t3_cold_ms}ms"
    result["T3_warm_ms"] = t3_warm_ms
    if result_warm:
        result["notes"] += f", warm={t3_warm_ms}ms"
    
    if result["status"] == "✓":
        result["notes"] += f"; latency={t3_warm_ms}ms"
    
    # T3/T4 failures are informational (don't affect exit code)
    if result_cold:
        result["T3_cold_ms"] = t3_cold_ms
        result["notes"] += f"; cold={t3_cold_ms}ms" if result["notes"] else f"cold={t3_cold_ms}ms"
    if result_warm:
        result["T3_warm_ms"] = t3_warm_ms
        result["notes"] += f", warm={t3_warm_ms}ms"
    
    return result


# ---------------------------------------------------------------------------
# T4 — OptionsProposal round-trip via wheel_schemas
# ---------------------------------------------------------------------------

async def _test_options_proposal(ep: dict, timeout: int, prompt: str,
                                  expected_action: WheelAction,
                                  expected_no_op_reason=None) -> dict:
    """Shared body for T4a and T4b.

    Uses include_raw=True so we can detect the llama.cpp Qwen3 mode where
    the server returns reasoning_content but empty content (which would
    otherwise silently produce a default-filled OptionsProposal that looks
    superficially valid). All field-level checks centralized in
    tradingagents.agents.wheel_schema_assertions so the assertion logic
    itself is unit-tested.
    """
    base_url = ep["base_url"].rstrip("/")
    test_name = "T4a" if expected_action == WheelAction.SELL_PUT else "T4b"
    result = {"test": test_name, "status": "—", "latency_ms": 0, "notes": ""}

    try:
        llm = ChatOpenAI(
            model=ep["model"],
            api_key=ep["api_key"],
            base_url=base_url,
            temperature=0,
            max_tokens=2048,  # structured output needs room for reasoning + complex JSON
        )
        # include_raw=True returns {"raw", "parsed", "parsing_error"} so we can
        # inspect the wire-level response before trusting the parsed object.
        # This is the only reliable way to catch empty-content / reasoning-only
        # responses from llama.cpp Qwen3 endpoints.
        llm_structured = llm.with_structured_output(OptionsProposal, include_raw=True)
    except Exception as e:
        result["status"] = "⊘"
        result["notes"] = f"binding_unsupported: {str(e)[:100]}"
        return result

    for attempt in range(2):
        try:
            start = time.monotonic()
            wrapped = asyncio.wait_for(llm_structured.ainvoke(prompt), timeout=timeout)
            raw_result = await wrapped
            elapsed = time.monotonic() - start
            result["latency_ms"] = round(elapsed * 1000)

            # Centralized assertion — checks raw content nonempty, parsing
            # succeeded, action matches, substantive fields populated,
            # rationale non-trivial. See wheel_schema_assertions.py for
            # the full check list and its unit tests.
            ok, note = assert_structured_response_complete(
                raw_result,
                expected_action=expected_action,
                expected_no_op_reason=expected_no_op_reason,
            )
            if ok:
                result["status"] = "✓"
                return result

            result["status"] = "✗"
            result["notes"] = note
            return result

        except asyncio.TimeoutError:
            result["status"] = "✗"
            result["notes"] = "timeout"
            return result  # don't retry timeouts
        except Exception as e:
            result["notes"] = str(e)[:160]
            if attempt == 0:
                # Retry once — may be transient parse error from truncated JSON
                continue
            result["status"] = "✗"
            return result
    return result


async def test_t4a_trade(ep: dict, timeout: int) -> dict:
    prompt = (
        "AAPL is trading at 175. The account has $20,000 cash, no AAPL position, "
        "no earnings within 14 days, regime is trend_up. "
        "Propose a cash-secured put: 0.20 delta, 30 DTE target."
    )
    return await _test_options_proposal(
        ep, timeout, prompt,
        expected_action=WheelAction.SELL_PUT,
    )


async def test_t4b_noop(ep: dict, timeout: int) -> dict:
    prompt = (
        "AAPL has confirmed earnings tomorrow. The account is in cash. "
        "Earnings blackout window is 7 days. Propose an action."
    )
    return await _test_options_proposal(
        ep, timeout, prompt,
        expected_action=WheelAction.NO_OP,
        expected_no_op_reason=NoOpReason.EARNINGS_BLACKOUT,
    )


# ---------------------------------------------------------------------------
# Run all tests for one endpoint (serial within cell)
# ---------------------------------------------------------------------------

async def run_cell(ep: dict, timeout: int) -> dict:
    results = {}

    # T1
    results["T1"] = await test_reachability(ep, timeout)
    if results["T1"]["status"] == "✗":
        results["T1"]["notes"] = "skipped" + ("; " + results["T1"]["notes"] if results["T1"]["notes"] else "")

    # T2 depends on T1
    if results["T1"]["status"] == "✓":
        results["T2"] = await test_plain_chat(ep, timeout)
        # T3 and T4 depend on T2
        if results["T2"]["status"] == "✓":
            results["T3"] = await test_structured_output(ep, timeout)
            results["T4a"] = await test_t4a_trade(ep, timeout)
            results["T4b"] = await test_t4b_noop(ep, timeout)
        else:
            for t in ("T3", "T4a", "T4b"):
                results[t] = {"test": t, "status": "—", "latency_ms": 0, "notes": "skipped (T2 failed)"}
    else:
        for t in ("T2", "T3", "T4a", "T4b"):
            results[t] = {"test": t, "status": "—", "latency_ms": 0, "notes": "skipped (T1 failed)"}

    return {"name": ep["name"], "results": results, "timestamp": datetime.utcnow().isoformat() + "Z"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    # Config at repo root (parent of scripts/)
    config_path = os.path.join(os.path.dirname(_HERE), "gx10_endpoints.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    endpoints = config["endpoints"]
    defaults = config.get("defaults", {})
    timeout = defaults.get("timeout_seconds", 60)
    max_parallel = defaults.get("max_parallel", 4)

    sem = asyncio.Semaphore(max_parallel)

    async def bounded(ep):
        async with sem:
            return await run_cell(ep, timeout)

    tasks = [bounded(ep) for ep in endpoints]
    cells = await asyncio.gather(*tasks)

    # ── ASCII table ──────────────────────────────────────────────────────
    test_cols = ["T1", "T2", "T3", "T4a", "T4b"]
    names = [c["name"] for c in cells]
    max_name = max(len(n) for n in names) if names else 10
    col_w = 4
    header = f"{'cell':<{max_name}} | {'T1':>{col_w}} | {'T2':>{col_w}} | {'T3':>{col_w}} | {'T4a':>{col_w}} | {'T4b':>{col_w}} | notes"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for c in cells:
        r = c["results"]
        row = f"{c['name']:<{max_name}}"
        for t in test_cols:
            v = r.get(t, {}).get("status", "—")
            row += f" | {v:>{col_w}}"
        notes = []
        for t in test_cols:
            n = r.get(t, {}).get("notes", "")
            if n and n != "—":
                notes.append(n)
        row += f" | {'; '.join(notes)[:60]}"
        print(row)
    print(sep)

    # ── JSON logging ─────────────────────────────────────────────────────
    log_dir = os.path.join(os.path.dirname(_HERE), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(log_dir, f"gx10_smoke_{timestamp}.json")

    with open(log_path, "w") as f:
        json.dump(cells, f, indent=2, default=str)
    print(f"\nLog: {log_path}")

    # ── Exit code ─────────────────────────────────────────────────────
    # Exit 0 if all cells pass T1 and T2; exit 1 otherwise.
    # T3/T4 failures are informational and don't affect exit code.
    t1_t2_ok = all(
        c["results"].get("T1", {}).get("status") == "✓" and
        c["results"].get("T2", {}).get("status") == "✓"
        for c in cells
    )
    return 0 if t1_t2_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
