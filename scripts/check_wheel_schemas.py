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

print()
if failures:
    print(f"FAILURES: {len(failures)}")
    sys.exit(1)
else:
    print("All schema tests passed.")
