"""Sanity-check the wheel schemas: valid cases pass, invalid cases raise."""

import sys
import os
import importlib.util

# Load wheel_schemas directly so we don't pull in langchain via agents/__init__.py
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCHEMA_PATH = os.path.join(_HERE, "..", "tradingagents", "agents", "wheel_schemas.py")
_spec = importlib.util.spec_from_file_location("wheel_schemas", _SCHEMA_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

from datetime import date, datetime, timedelta
from pydantic import ValidationError

_names = (
    "OptionsProposal", "WheelDecision", "WheelState", "CycleOutcome",
    "OpenOptionPosition", "ShareLot", "RiskGateResult",
    "WheelAction", "CycleStage", "OptionRight", "NoOpReason", "RollReason", "CycleOutcomeStatus",
    "render_options_proposal", "render_wheel_decision", "render_cycle_outcome",
)
for _n in _names:
    globals()[_n] = getattr(_mod, _n)

failures = []

def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failures.append((name, e))


# 1. Valid SELL_PUT proposal
def valid_sto_put():
    p = OptionsProposal(
        symbol="AAPL",
        cycle_stage=CycleStage.CASH,
        action=WheelAction.SELL_PUT,
        right=OptionRight.PUT,
        strike=170.0,
        expiry=date.today() + timedelta(days=30),
        contracts=1,
        delta=-0.20,
        expected_premium=1.85,
        dte=30,
        rationale="0.20-delta put 30 DTE, premium meets target, regime favorable.",
    )
    assert p.action == WheelAction.SELL_PUT
    md = render_options_proposal(p)
    assert "sell_cash_secured_put" in md, f"expected new wire-format value, got: {md}"

# 2. Valid NO_OP with structured reason
def valid_no_op():
    p = OptionsProposal(
        symbol="TSLA",
        cycle_stage=CycleStage.CASH,
        action=WheelAction.NO_OP,
        no_op_reason=NoOpReason.EARNINGS_BLACKOUT,
        rationale="Earnings 2026-05-12, within 7d blackout window.",
    )
    assert p.no_op_reason == NoOpReason.EARNINGS_BLACKOUT

# 3. NO_OP without reason must fail (the silent-failure regression guardrail)
def no_op_without_reason_fails():
    try:
        OptionsProposal(
            symbol="X", cycle_stage=CycleStage.CASH,
            action=WheelAction.NO_OP, rationale="...",
        )
        raise AssertionError("should have raised")
    except ValidationError as e:
        assert "no_op_reason" in str(e)

# 4. Trade action missing strike must fail
def sto_put_missing_strike_fails():
    try:
        OptionsProposal(
            symbol="X", cycle_stage=CycleStage.CASH,
            action=WheelAction.SELL_PUT,
            right=OptionRight.PUT,
            expiry=date.today() + timedelta(days=30),
            contracts=1,
            rationale="...",
        )
        raise AssertionError("should have raised")
    except ValidationError as e:
        assert "strike" in str(e)

# 5. Right/action mismatch must fail
def call_action_with_put_right_fails():
    try:
        OptionsProposal(
            symbol="X", cycle_stage=CycleStage.ASSIGNED,
            action=WheelAction.SELL_CALL,
            right=OptionRight.PUT,  # WRONG
            strike=100, expiry=date.today() + timedelta(days=30),
            contracts=1, rationale="...",
        )
        raise AssertionError("should have raised")
    except ValidationError as e:
        err = str(e).lower()
        assert "call" in err or "put" in err, f"expected put/call in error: {e}"

# 6. ROLL_PUT requires roll_reason and closing premium
def roll_put_missing_metadata_fails():
    try:
        OptionsProposal(
            symbol="X", cycle_stage=CycleStage.SHORT_PUT,
            action=WheelAction.ROLL_PUT,
            right=OptionRight.PUT,
            strike=95, expiry=date.today() + timedelta(days=45),
            contracts=1, rationale="...",
        )
        raise AssertionError("should have raised")
    except ValidationError as e:
        assert "roll" in str(e).lower()

# 7. WheelState stage detection
def stage_detection():
    state = WheelState(
        as_of=datetime.now(),
        cash_available=10000, buying_power=20000, net_liquidation=50000,
        share_lots=[ShareLot(symbol="AAPL", quantity=100, cost_basis=175.0, acquired_date=date.today())],
    )
    assert state.stage_for("AAPL") == CycleStage.ASSIGNED
    assert state.stage_for("TSLA") == CycleStage.CASH

# 8. OpenOptionPosition DTE and profit_pct
def position_helpers():
    pos = OpenOptionPosition(
        symbol="AAPL", right=OptionRight.PUT, strike=170,
        expiry=date.today() + timedelta(days=14),
        contracts=1, open_premium=2.00, current_mid=0.50,
        open_date=date.today() - timedelta(days=16),
    )
    assert pos.dte == 14
    assert abs(pos.profit_pct - 0.75) < 1e-9  # 75% of premium captured

# 9. WheelDecision render is valid markdown
def decision_render():
    proposal = OptionsProposal(
        symbol="AAPL", cycle_stage=CycleStage.CASH,
        action=WheelAction.SELL_PUT, right=OptionRight.PUT,
        strike=170.0, expiry=date.today() + timedelta(days=30),
        contracts=1, delta=-0.20, expected_premium=1.85, dte=30,
        rationale="Fits criteria.",
    )
    decision = WheelDecision(
        selected_proposal=proposal,
        executive_summary="AAPL beat alternatives on premium-to-risk ratio.",
        regime_at_decision="trend_up",
        risk_gates=[
            RiskGateResult(name="earnings_blackout", passed=True),
            RiskGateResult(name="buying_power", passed=True, detail="$18,500 BP, $17,000 required"),
        ],
    )
    md = render_wheel_decision(decision)
    assert "FINAL WHEEL ACTION" in md
    assert "sell_cash_secured_put" in md, f"expected new wire value in: {md}"

# 10. CycleOutcome record
def cycle_outcome():
    o = CycleOutcome(
        decision_id="2026-05-10-AAPL-1",
        symbol="AAPL", action_taken=WheelAction.SELL_PUT,
        open_date=date(2026, 4, 1), close_date=date(2026, 5, 1),
        status=CycleOutcomeStatus.EXPIRED_WORTHLESS,
        premium_collected=185.0, capital_at_risk=17000.0,
        realized_return_pct=0.0109, days_held=30,
        regime_at_open="trend_up",
        setup_signature="put_delta20_dte30_trend_up",
    )
    md = render_cycle_outcome(o)
    assert "expired_worthless" in md, f"expected new wire value in: {md}"


for name, fn in [
    ("valid SELL_PUT", valid_sto_put),
    ("valid NO_OP with reason", valid_no_op),
    ("NO_OP without reason fails", no_op_without_reason_fails),
    ("SELL_PUT without strike fails", sto_put_missing_strike_fails),
    ("CALL action + PUT right fails", call_action_with_put_right_fails),
    ("ROLL_PUT without metadata fails", roll_put_missing_metadata_fails),
    ("WheelState.stage_for", stage_detection),
    ("OpenOptionPosition helpers", position_helpers),
    ("WheelDecision render", decision_render),
    ("CycleOutcome render", cycle_outcome),
]:
    check(name, fn)


# ---------------------------------------------------------------------------
# Negative tests for wheel_schema_assertions
# ---------------------------------------------------------------------------
# These prove the assertion helpers reject the failure modes we've actually
# seen from the GX10 cluster (empty content from llama.cpp Qwen3, garbage
# responses from misconfigured vLLM, default-filled objects from langchain
# fallback paths). Every negative test corresponds to a real diagnostic
# event documented in docs/LATENCY_ANALYSIS_2026-05-11.md or the smoke logs.
# ---------------------------------------------------------------------------

# Load wheel_schema_assertions the same way as wheel_schemas (no langchain init)
_ASSERTION_PATH = os.path.join(_HERE, "..", "tradingagents", "agents", "wheel_schema_assertions.py")
_aspec = importlib.util.spec_from_file_location("wheel_schema_assertions", _ASSERTION_PATH)
_amod = importlib.util.module_from_spec(_aspec)
# wheel_schema_assertions imports from tradingagents.agents.wheel_schemas
# Inject our pre-loaded schemas module under that name to avoid the package init
import sys as _sys
_sys.modules["tradingagents"] = type(_sys)("tradingagents")
_sys.modules["tradingagents.agents"] = type(_sys)("tradingagents.agents")
_sys.modules["tradingagents.agents.wheel_schemas"] = _mod
_aspec.loader.exec_module(_amod)

assert_raw_content_nonempty = _amod.assert_raw_content_nonempty
assert_proposal_not_none = _amod.assert_proposal_not_none
assert_rationale_substantive = _amod.assert_rationale_substantive
assert_valid_sell_put = _amod.assert_valid_sell_put
assert_valid_no_op = _amod.assert_valid_no_op
assert_structured_response_complete = _amod.assert_structured_response_complete


# Mock raw-response shape that mirrors langchain's AIMessage
class _MockRaw:
    def __init__(self, content="", reasoning_content=""):
        self.content = content
        self.additional_kwargs = {"reasoning_content": reasoning_content} if reasoning_content else {}


# 11. assert_raw_content_nonempty rejects None
def raw_none_rejected():
    ok, note = assert_raw_content_nonempty(None)
    assert not ok and "None" in note, f"expected rejection, got ({ok}, {note})"

# 12. assert_raw_content_nonempty rejects empty content
def raw_empty_rejected():
    ok, note = assert_raw_content_nonempty(_MockRaw(content=""))
    assert not ok, f"expected rejection of empty content, got ({ok}, {note})"

# 13. assert_raw_content_nonempty surfaces llama.cpp reasoning_content issue
#     (the actual May-2026 diagnostic from Rivet — content empty, reasoning populated)
def raw_reasoning_only_diagnostic():
    ok, note = assert_raw_content_nonempty(
        _MockRaw(content="", reasoning_content="Let me think... the user wants a put...")
    )
    assert not ok, "should reject"
    assert "reasoning_content" in note, f"diagnostic should mention reasoning_content: {note}"

# 14. assert_raw_content_nonempty accepts populated content
def raw_content_accepted():
    ok, note = assert_raw_content_nonempty(_MockRaw(content="some real response"))
    assert ok, f"should accept populated content: {note}"

# 15. assert_proposal_not_none rejects None
def proposal_none_rejected():
    ok, note = assert_proposal_not_none(None)
    assert not ok and "None" in note, f"expected rejection: ({ok}, {note})"

# 16. assert_proposal_not_none rejects wrong type
def proposal_wrong_type_rejected():
    ok, note = assert_proposal_not_none({"action": "something"})
    assert not ok, f"should reject dict: ({ok}, {note})"

# 17. assert_rationale_substantive rejects empty/short rationale
#     (a model that returns rationale="ok" satisfies the schema but is useless)
def rationale_empty_rejected():
    p = OptionsProposal(
        symbol="AAPL", cycle_stage=CycleStage.CASH,
        action=WheelAction.SELL_PUT, right=OptionRight.PUT,
        strike=170.0, expiry=date.today() + timedelta(days=30),
        contracts=1, rationale="ok",
    )
    ok, note = assert_rationale_substantive(p)
    assert not ok, f"should reject short rationale: ({ok}, {note})"
    assert "too short" in note

# 18. assert_valid_sell_put catches null strike
def sell_put_null_strike_caught():
    # Use bypass-validator construction since Pydantic would normally reject this;
    # we want to simulate what happens when langchain returns a partially-populated
    # OptionsProposal (e.g. from a fallback path)
    try:
        p = OptionsProposal(
            symbol="AAPL", cycle_stage=CycleStage.CASH,
            action=WheelAction.SELL_PUT, right=OptionRight.PUT,
            expiry=date.today() + timedelta(days=30),
            contracts=1, rationale="A meaningful rationale for the trade decision.",
        )
        # Pydantic actually rejects this because strike is required for SELL_PUT
        raise AssertionError("Pydantic should have rejected null strike")
    except ValidationError:
        # Good — Pydantic catches this at construction. Test the assertion separately
        # by constructing a model_construct (bypass validation) and asserting our
        # helper catches what slipped through.
        p = OptionsProposal.model_construct(
            symbol="AAPL", cycle_stage=CycleStage.CASH,
            action=WheelAction.SELL_PUT, right=OptionRight.PUT,
            strike=None,  # null — should be caught by assertion
            expiry=date.today() + timedelta(days=30),
            contracts=1, rationale="A meaningful rationale.",
        )
        ok, note = assert_valid_sell_put(p)
        assert not ok and "strike" in note, f"expected strike rejection: ({ok}, {note})"

# 19. assert_valid_sell_put accepts a well-formed proposal
def sell_put_well_formed_accepted():
    p = OptionsProposal(
        symbol="AAPL", cycle_stage=CycleStage.CASH,
        action=WheelAction.SELL_PUT, right=OptionRight.PUT,
        strike=170.0, expiry=date.today() + timedelta(days=30),
        contracts=1, delta=-0.20, expected_premium=1.85, dte=30,
        rationale="Selling a 0.20-delta 30-DTE put on AAPL at 170 strike for $1.85 premium.",
    )
    ok, note = assert_valid_sell_put(p)
    assert ok, f"well-formed proposal should pass: {note}"

# 20. assert_valid_no_op rejects mismatched reason
def no_op_wrong_reason_caught():
    p = OptionsProposal(
        symbol="AAPL", cycle_stage=CycleStage.CASH,
        action=WheelAction.NO_OP, no_op_reason=NoOpReason.REGIME_UNFAVORABLE,
        rationale="Market regime risk_off; standing down per regime gate.",
    )
    ok, note = assert_valid_no_op(p, expected_reason=NoOpReason.EARNINGS_BLACKOUT)
    assert not ok and "no_op_reason" in note, f"expected reason mismatch: ({ok}, {note})"

# 21. assert_valid_no_op accepts a well-formed NO_OP proposal
def no_op_well_formed_accepted():
    p = OptionsProposal(
        symbol="AAPL", cycle_stage=CycleStage.CASH,
        action=WheelAction.NO_OP, no_op_reason=NoOpReason.EARNINGS_BLACKOUT,
        rationale="AAPL has confirmed earnings tomorrow, within 7-day blackout window.",
    )
    ok, note = assert_valid_no_op(p, expected_reason=NoOpReason.EARNINGS_BLACKOUT)
    assert ok, f"well-formed NO_OP should pass: {note}"

# 22. assert_structured_response_complete — end-to-end happy path
def structured_response_happy_path():
    p = OptionsProposal(
        symbol="AAPL", cycle_stage=CycleStage.CASH,
        action=WheelAction.SELL_PUT, right=OptionRight.PUT,
        strike=170.0, expiry=date.today() + timedelta(days=30),
        contracts=1, rationale="A 0.20-delta 30-DTE cash-secured put on AAPL.",
    )
    result = {
        "raw": _MockRaw(content='{"action": "sell_cash_secured_put", ...}'),
        "parsed": p,
        "parsing_error": None,
    }
    ok, note = assert_structured_response_complete(result, expected_action=WheelAction.SELL_PUT)
    assert ok, f"happy path should pass: {note}"

# 23. assert_structured_response_complete — catches parsing_error
def structured_response_catches_parse_error():
    result = {
        "raw": _MockRaw(content="garbage"),
        "parsed": None,
        "parsing_error": ValueError("could not parse"),
    }
    ok, note = assert_structured_response_complete(result, expected_action=WheelAction.SELL_PUT)
    assert not ok and "parsing error" in note, f"should catch parse error: ({ok}, {note})"

# 24. assert_structured_response_complete — catches the real llama.cpp Qwen3 mode
#     (empty content + reasoning_content populated; this is the actual May-2026 failure)
def structured_response_catches_reasoning_only():
    result = {
        "raw": _MockRaw(content="", reasoning_content="Let me think about this..."),
        "parsed": None,  # langchain often returns None when content is empty
        "parsing_error": None,
    }
    ok, note = assert_structured_response_complete(result, expected_action=WheelAction.SELL_PUT)
    assert not ok, f"should reject empty-content reasoning-only response: ({ok}, {note})"
    assert "reasoning_content" in note or "empty" in note, \
        f"diagnostic should explain the issue: {note}"


for name, fn in [
    ("raw None rejected", raw_none_rejected),
    ("raw empty content rejected", raw_empty_rejected),
    ("raw reasoning_content-only diagnostic surfaced", raw_reasoning_only_diagnostic),
    ("raw populated content accepted", raw_content_accepted),
    ("proposal None rejected", proposal_none_rejected),
    ("proposal wrong type rejected", proposal_wrong_type_rejected),
    ("short rationale rejected", rationale_empty_rejected),
    ("SELL_PUT null strike caught by assertion", sell_put_null_strike_caught),
    ("well-formed SELL_PUT accepted", sell_put_well_formed_accepted),
    ("NO_OP wrong reason caught", no_op_wrong_reason_caught),
    ("well-formed NO_OP accepted", no_op_well_formed_accepted),
    ("structured response happy path", structured_response_happy_path),
    ("structured response catches parse error", structured_response_catches_parse_error),
    ("structured response catches reasoning-only (real llama.cpp Qwen3 bug)", structured_response_catches_reasoning_only),
]:
    check(name, fn)


print()
if failures:
    print(f"FAILURES: {len(failures)}")
    sys.exit(1)
else:
    print("All schema tests passed.")
