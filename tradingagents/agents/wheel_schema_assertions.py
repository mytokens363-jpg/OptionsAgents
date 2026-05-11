"""Assertion helpers for validating LLM-generated OptionsProposal instances.

The smoke test and any future agent-output tests use these helpers instead of
inline assertions. Centralizing the validation logic here means:

1. The assertion rules themselves can be unit-tested (negative tests confirm
   that empty/default/garbage proposals are rejected).
2. Future test suites — Stage 3 agent-graph tests, Stage 5 shadow-mode
   regression tests — reuse the same hardened checks, so a model that
   sneaks past validation in one place will be caught in all places.
3. The "what counts as a valid proposal" definition lives in one file,
   versioned in git, with the schema it tests.

Design lessons baked in:

- **Test the harness.** Multiple GX10 smoke runs produced superficially-pass
  results from models that were actually emitting empty or default-filled
  output. Every assertion below has a corresponding negative test in
  scripts/check_wheel_schemas.py that proves the assertion rejects garbage.

- **Substantive over structural.** Pydantic + model_validator already ensure
  the schema's *structure* is satisfied. These helpers go further: they check
  that the model populated the fields with *meaningful* values, not just
  Pydantic-acceptable nulls. A rationale of three characters validates
  structurally but is not a real rationale.

- **No string-value comparisons.** All comparisons use enum members
  (`WheelAction.SELL_PUT`), never wire-format strings. This insulates the
  test logic from any future enum-value redesign (like the 2026-05-10
  STO_PUT → sell_cash_secured_put migration).
"""

from typing import Optional, Tuple

from tradingagents.agents.wheel_schemas import (
    CycleStage,
    NoOpReason,
    OptionRight,
    OptionsProposal,
    WheelAction,
)


# Type for assertion results: (passed, failure_note)
AssertResult = Tuple[bool, str]


def assert_raw_content_nonempty(raw_response) -> AssertResult:
    """Verify the raw LLM response carries non-empty content.

    Catches the llama.cpp Qwen3 issue where the server returns
    `reasoning_content` populated but `content` empty. Without this check,
    structured-output binding silently fails downstream and we get
    misleading test results.

    Pass `raw_response` as the `.raw` field from
    `llm.with_structured_output(Schema, include_raw=True).invoke(prompt)`.
    Or any object with a `.content` attribute (e.g. an AIMessage).
    """
    if raw_response is None:
        return False, "raw response is None"
    content = getattr(raw_response, "content", None)
    if content is None:
        return False, "raw response has no .content attribute"
    if not isinstance(content, str):
        # langchain sometimes returns list-of-blocks; coerce defensively
        try:
            content = str(content)
        except Exception:
            return False, f"raw content not coercible to str: type={type(content).__name__}"
    if not content.strip():
        # Common llama.cpp Qwen3 path: reasoning_content populated, content empty
        reasoning = getattr(raw_response, "additional_kwargs", {}).get("reasoning_content", "")
        if reasoning:
            return False, f"empty content; reasoning_content has {len(reasoning)} chars (server returning reasoning in wrong field)"
        return False, "empty content"
    return True, ""


def assert_proposal_not_none(proposal: Optional[OptionsProposal]) -> AssertResult:
    """First gate — did structured parsing produce anything at all?"""
    if proposal is None:
        return False, "proposal is None (structured-output binding returned nothing)"
    if not isinstance(proposal, OptionsProposal):
        return False, f"proposal is not an OptionsProposal: type={type(proposal).__name__}"
    return True, ""


def assert_rationale_substantive(proposal: OptionsProposal, min_chars: int = 20) -> AssertResult:
    """Rationale must be non-trivial prose, not a single word or whitespace.

    A model that returns rationale="ok" satisfies the schema's str type but
    has not actually rationalized anything. Stage 3's reflection loop needs
    real rationale text to key memory retrieval on, so we enforce it here.
    """
    r = (proposal.rationale or "").strip()
    if len(r) < min_chars:
        return False, f"rationale too short ({len(r)} chars, min {min_chars}): {r!r}"
    return True, ""


def assert_valid_sell_put(proposal: OptionsProposal) -> AssertResult:
    """All substantive checks for a SELL_PUT proposal.

    Used by smoke test T4a and any future analyst-output test where the
    expected action is sell_cash_secured_put.
    """
    ok, note = assert_proposal_not_none(proposal)
    if not ok:
        return False, note

    checks = []
    if proposal.action != WheelAction.SELL_PUT:
        checks.append(
            f"action={proposal.action.value} ({proposal.action.name}), "
            f"expected {WheelAction.SELL_PUT.value} ({WheelAction.SELL_PUT.name})"
        )
    if proposal.right != OptionRight.PUT:
        right_val = proposal.right.value if proposal.right else None
        checks.append(f"right={right_val}, expected {OptionRight.PUT.value}")
    if proposal.cycle_stage != CycleStage.CASH:
        checks.append(f"cycle_stage={proposal.cycle_stage.value}, expected {CycleStage.CASH.value}")
    if proposal.strike is None or proposal.strike <= 0:
        checks.append(f"strike={proposal.strike} (must be > 0)")
    if proposal.expiry is None:
        checks.append("expiry is null")
    if proposal.contracts is None or proposal.contracts <= 0:
        checks.append(f"contracts={proposal.contracts} (must be > 0)")

    # Rationale check
    rat_ok, rat_note = assert_rationale_substantive(proposal)
    if not rat_ok:
        checks.append(rat_note)

    if checks:
        return False, "; ".join(checks)
    return True, ""


def assert_valid_no_op(
    proposal: OptionsProposal,
    expected_reason: NoOpReason,
) -> AssertResult:
    """All substantive checks for a NO_OP proposal.

    Used by smoke test T4b and any future test where the expected action is
    do_not_trade due to a risk gate.
    """
    ok, note = assert_proposal_not_none(proposal)
    if not ok:
        return False, note

    checks = []
    if proposal.action != WheelAction.NO_OP:
        checks.append(
            f"action={proposal.action.value} ({proposal.action.name}), "
            f"expected {WheelAction.NO_OP.value} ({WheelAction.NO_OP.name})"
        )
    if proposal.no_op_reason != expected_reason:
        actual = proposal.no_op_reason.value if proposal.no_op_reason else None
        checks.append(f"no_op_reason={actual}, expected {expected_reason.value}")

    rat_ok, rat_note = assert_rationale_substantive(proposal)
    if not rat_ok:
        checks.append(rat_note)

    if checks:
        return False, "; ".join(checks)
    return True, ""


def assert_structured_response_complete(
    result: dict,
    expected_action: WheelAction,
    expected_no_op_reason: Optional[NoOpReason] = None,
) -> AssertResult:
    """End-to-end assertion for a `with_structured_output(..., include_raw=True)` result.

    Verifies in order:
    1. There was no parsing error
    2. The raw response had non-empty content
    3. A proposal was returned (not None)
    4. The proposal's action and other substantive fields match expectations

    This is the assertion smoke test cells should use. It catches every
    failure mode we've seen so far at the right layer.
    """
    if not isinstance(result, dict):
        return False, f"result is not a dict (did you forget include_raw=True?): type={type(result).__name__}"

    parsing_error = result.get("parsing_error")
    if parsing_error is not None:
        return False, f"structured-output parsing error: {str(parsing_error)[:200]}"

    raw = result.get("raw")
    raw_ok, raw_note = assert_raw_content_nonempty(raw)
    if not raw_ok:
        return False, f"raw response: {raw_note}"

    parsed = result.get("parsed")
    if expected_action == WheelAction.NO_OP:
        assert expected_no_op_reason is not None, "must supply expected_no_op_reason for NO_OP assertions"
        return assert_valid_no_op(parsed, expected_no_op_reason)
    elif expected_action == WheelAction.SELL_PUT:
        return assert_valid_sell_put(parsed)
    else:
        # Generic structural check for other actions; can be expanded as Stage 3 tests need
        ok, note = assert_proposal_not_none(parsed)
        if not ok:
            return False, note
        if parsed.action != expected_action:
            return False, f"action={parsed.action.value}, expected {expected_action.value}"
        return assert_rationale_substantive(parsed)
