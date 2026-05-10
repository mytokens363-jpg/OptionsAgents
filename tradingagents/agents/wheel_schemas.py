"""Pydantic schemas for OptionsAgents — wheel strategy decision artifacts.

Mirrors the structure of ``tradingagents/agents/schemas.py`` (TraderProposal,
PortfolioDecision) but encodes wheel-strategy semantics:

- ``OptionsProposal``: per-symbol options trade proposal from an analyst agent
- ``WheelState``: read-only snapshot of account + open positions + cycle stage
- ``WheelDecision``: final action emitted by the Portfolio Manager analogue
- ``CycleOutcome``: realized outcome record used by the reflection/memory loop

Design principles carried over from the WheelBot debug experience:

- **Typed failure reasons**: where TradingAgents lets an analyst return prose,
  wheel decisions that *cannot* proceed return an explicit ``WheelAction.NO_OP``
  with a structured ``no_op_reason``. The April 2026 scanner regression was
  masked for weeks by a silent empty-list return; the schema layer encodes
  that lesson — there is no "silent no" in this system.
- **Cycle-stage awareness**: every proposal carries the cycle stage the
  account is currently in (CASH / SHORT_PUT / ASSIGNED / SHORT_CALL). The
  same chain analysis means different things depending on whether the agent
  is opening a put, rolling a put, or selling a covered call against assigned
  shares.
- **State files are authoritative**: ``WheelState`` is constructed from live
  IBKR data + the state-file ledger (``trades.jsonl``, ``rolls.jsonl``,
  ``regime.json``, ``earnings_dates.json``). The agent never re-derives state
  it can read from the source of truth.

The render helpers turn parsed instances back into the markdown shape the
existing TradingAgents memory log and CLI display already consume.
"""

from datetime import date, datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OptionRight(str, Enum):
    """Option contract right."""

    # Plain-English values; single letters ("P", "C") gave models no semantic
    # anchor in GX10 smoke tests. The 122B got these right but smaller models
    # may not — defending consistently across model sizes.
    PUT = "put"
    CALL = "call"


class WheelAction(str, Enum):
    """All actions an OptionsAgents proposal can take.

    The wheel has four mechanical legs plus an explicit no-op. Rolls are
    represented as compound actions because the executor needs to know it's
    a single net-credit transaction, not two independent legs.
    """

    # IMPORTANT — wire-format value design note:
    # The string values below are what the LLM sees and must select from when
    # filling a structured-output schema. We deliberately use plain-English
    # phrases ("sell_cash_secured_put") rather than trader jargon ("STO_PUT")
    # because GX10 smoke-testing in May 2026 showed that all four tested
    # models (Qwen3.6-35B, Qwen3.5-35B, Qwen3.5-122B, gemma4:26b) defaulted
    # to "STO_CALL" when asked to propose a cash-secured put — they were
    # correctly identifying the option right but incorrectly picking the
    # action enum, because terse jargon values gave them no semantic anchor
    # to map "cash-secured put" onto. Plain-English values let the model
    # match the prompt's vocabulary directly to an allowed value.
    # The Python *names* (SELL_PUT, NO_OP, etc.) stay stable — no executor,
    # memory log, or downstream consumer needs to change.
    SELL_PUT = "sell_cash_secured_put"
    BUY_PUT = "buy_put_to_close"
    SELL_CALL = "sell_covered_call"
    BUY_CALL = "buy_call_to_close"
    ROLL_PUT = "roll_put_to_later_expiry"
    ROLL_CALL = "roll_call_to_later_expiry"
    NO_OP = "do_not_trade"


class CycleStage(str, Enum):
    """Where this symbol currently sits in the wheel cycle."""

    # Plain-English values for the same reason as WheelAction (above).
    CASH = "cash_no_position"          # No position; eligible to sell puts
    SHORT_PUT = "short_put_open"       # Open short put; awaiting expiry or assignment
    ASSIGNED = "shares_assigned"       # Shares assigned; eligible to sell calls
    SHORT_CALL = "covered_call_open"   # Open covered call; awaiting expiry or called-away


class NoOpReason(str, Enum):
    """Structured reasons for declining to act. Never silent."""

    # Values are plain-English so the model can map prompt vocabulary
    # ("earnings tomorrow", "no buying power") directly to an allowed reason.
    EARNINGS_BLACKOUT = "earnings_blackout_window"
    REGIME_UNFAVORABLE = "market_regime_unfavorable"
    CHAIN_UNAVAILABLE = "option_chain_unavailable"
    NO_VIABLE_STRIKE = "no_strike_meets_criteria"
    BUYING_POWER_INSUFFICIENT = "buying_power_insufficient"
    CONCENTRATION_LIMIT = "position_concentration_limit_exceeded"
    POSITION_AT_PROFIT_TARGET = "position_at_profit_target_hold_to_close"
    DTE_OUTSIDE_BAND = "no_expiry_in_target_window"
    DUPLICATE_PROPOSAL = "duplicate_position_already_open"
    OTHER = "other_see_rationale"


class RollReason(str, Enum):
    """Why a roll is being proposed (for memory/reflection tagging)."""

    DEFENSIVE_ITM = "defensive_strike_breached"
    PROFIT_TAKE_AND_EXTEND = "profit_target_reached_extend_position"
    DTE_MANAGEMENT = "approaching_expiry_no_other_reason"


# ---------------------------------------------------------------------------
# Wheel state snapshot
# ---------------------------------------------------------------------------


class OpenOptionPosition(BaseModel):
    """A single open short option position."""

    symbol: str
    right: OptionRight
    strike: float
    expiry: date
    contracts: int = Field(description="Always positive; short positions are implicit by short_*= true")
    open_premium: float = Field(description="Premium received per share at open (positive)")
    current_mid: Optional[float] = Field(default=None, description="Current mid price per share, if available")
    open_date: date

    @property
    def dte(self) -> int:
        """Days to expiry from today (UTC date)."""
        return (self.expiry - date.today()).days

    @property
    def profit_pct(self) -> Optional[float]:
        """Fraction of premium captured. Positive = profitable for the seller."""
        if self.current_mid is None or self.open_premium <= 0:
            return None
        return 1.0 - (self.current_mid / self.open_premium)


class ShareLot(BaseModel):
    """Shares held against which calls can be written."""

    symbol: str
    quantity: int
    cost_basis: float = Field(description="Effective per-share cost basis (incl. premiums credited)")
    acquired_date: date


class WheelState(BaseModel):
    """Read-only snapshot of account state the agents reason over.

    Populated by the IBKR + state-file tools at the start of each run.
    Treat as immutable within a single graph execution.
    """

    as_of: datetime
    cash_available: float = Field(description="Settled cash available for new cash-secured puts")
    buying_power: float = Field(description="Total buying power (may exceed cash on margin)")
    net_liquidation: float
    open_puts: List[OpenOptionPosition] = Field(default_factory=list)
    open_calls: List[OpenOptionPosition] = Field(default_factory=list)
    share_lots: List[ShareLot] = Field(default_factory=list)
    regime: Optional[str] = Field(default=None, description="From regime.json — e.g. 'trend_up', 'chop', 'risk_off'")
    pending_earnings: dict = Field(
        default_factory=dict,
        description="symbol -> ISO date of next earnings, from earnings_dates.json",
    )

    def stage_for(self, symbol: str) -> CycleStage:
        """Determine the cycle stage for a given symbol based on positions."""
        has_put = any(p.symbol == symbol for p in self.open_puts)
        has_call = any(c.symbol == symbol for c in self.open_calls)
        has_shares = any(s.symbol == symbol and s.quantity >= 100 for s in self.share_lots)

        if has_call:
            return CycleStage.SHORT_CALL
        if has_shares:
            return CycleStage.ASSIGNED
        if has_put:
            return CycleStage.SHORT_PUT
        return CycleStage.CASH


# ---------------------------------------------------------------------------
# Analyst proposal — what an OptionsAnalyst agent emits per candidate
# ---------------------------------------------------------------------------


class OptionsProposal(BaseModel):
    """Per-symbol options trade proposal from an analyst agent.

    The analyst evaluates the option chain in the context of the current
    cycle stage and emits exactly one proposal per symbol — including
    NO_OP proposals with a structured reason. The Portfolio Manager
    analogue then selects across competing proposals.
    """

    symbol: str = Field(description="Underlying ticker symbol, e.g. AAPL")
    cycle_stage: CycleStage = Field(
        description=(
            "The wheel cycle stage this symbol is in right now, derived from current positions. "
            "Use cash_no_position when there is no position and you can sell puts. "
            "Use short_put_open when there is already an open short put. "
            "Use shares_assigned when shares are held and you can sell covered calls. "
            "Use covered_call_open when there is an open covered call."
        ),
    )
    action: WheelAction = Field(
        description=(
            "The wheel action to take. Map the user's natural-language intent to the closest value below:\n"
            "- If the user wants to sell a cash-secured put (open a new short put for premium): "
            "use sell_cash_secured_put.\n"
            "- If the user wants to write a covered call against assigned shares: use sell_covered_call.\n"
            "- If the user wants to close an existing short put: use buy_put_to_close.\n"
            "- If the user wants to close an existing short call: use buy_call_to_close.\n"
            "- If the user wants to roll a put to a later expiry: use roll_put_to_later_expiry.\n"
            "- If the user wants to roll a call to a later expiry: use roll_call_to_later_expiry.\n"
            "- If risk gates (earnings within blackout window, unfavorable market regime, "
            "insufficient buying power, etc.) make any trade inappropriate, use do_not_trade and "
            "populate no_op_reason with the specific reason."
        ),
    )

    # Trade economics — required when action is not do_not_trade
    right: Optional[OptionRight] = Field(
        default=None,
        description=(
            "Option right: put or call. Must match the action — put actions require put, "
            "call actions require call. Omit only when action is do_not_trade."
        ),
    )
    strike: Optional[float] = Field(
        default=None,
        description="Strike price in dollars. Omit only when action is do_not_trade.",
    )
    expiry: Optional[date] = Field(
        default=None,
        description="Contract expiry date in ISO format (YYYY-MM-DD). Omit only when action is do_not_trade.",
    )
    contracts: Optional[int] = Field(
        default=None,
        description="Number of contracts; positive integer. Omit only when action is do_not_trade.",
    )
    delta: Optional[float] = Field(default=None, description="Option delta at proposal time (signed)")
    expected_premium: Optional[float] = Field(default=None, description="Expected mid-price premium per share")
    dte: Optional[int] = Field(default=None, description="Days to expiry from today")

    # Roll-specific
    roll_reason: Optional[RollReason] = Field(
        default=None,
        description=(
            "REQUIRED when action is roll_put_to_later_expiry or roll_call_to_later_expiry. "
            "Encodes why the roll is being proposed."
        ),
    )
    closing_position_premium: Optional[float] = Field(
        default=None,
        description=(
            "Premium to pay (per share) to close the existing leg before the new leg is opened. "
            "REQUIRED when action is a roll."
        ),
    )

    # do_not_trade path
    no_op_reason: Optional[NoOpReason] = Field(
        default=None,
        description=(
            "REQUIRED when action is do_not_trade. Encodes the specific risk gate or condition "
            "that prevents trading. Map the situation to the closest reason: earnings within "
            "blackout window → earnings_blackout_window; market regime blocks new opens → "
            "market_regime_unfavorable; no option chain data available → option_chain_unavailable; "
            "no strike meets delta/premium criteria → no_strike_meets_criteria; insufficient "
            "buying power → buying_power_insufficient; would exceed per-symbol concentration cap "
            "→ position_concentration_limit_exceeded; position is at profit target and should be "
            "closed not rolled → position_at_profit_target_hold_to_close; no expiry available in "
            "the target DTE window → no_expiry_in_target_window; already have an open position "
            "matching this proposal → duplicate_position_already_open; anything else → "
            "other_see_rationale with details in rationale field."
        ),
    )

    # Always required — natural language anchored in the analysis
    rationale: str = Field(
        description=(
            "Two to four sentences. For trade actions: justify strike, expiry, and "
            "sizing against the chain and the cycle stage. For do_not_trade: explain "
            "the specific evidence behind no_op_reason."
        ),
    )

    @model_validator(mode="after")
    def _validate_action_consistency(self):
        """Enforce that schema fields are consistent with the chosen action.

        do_not_trade must carry no_op_reason; trade actions must carry the legs;
        roll actions must carry roll_reason and closing_position_premium. This
        is the structural answer to the silent-failure regression — the
        proposal cannot validate as 'I won't do anything' without saying why.
        """
        if self.action == WheelAction.NO_OP:
            if self.no_op_reason is None:
                raise ValueError(
                    f"{self.action.value} proposals must include no_op_reason"
                )
            return self

        # All non-do_not_trade actions require the core trade legs
        missing = [
            name for name, value in (
                ("right", self.right),
                ("strike", self.strike),
                ("expiry", self.expiry),
                ("contracts", self.contracts),
            ) if value is None
        ]
        if missing:
            raise ValueError(
                f"{self.action.value} proposals must include: {', '.join(missing)}"
            )

        if self.action in (WheelAction.ROLL_PUT, WheelAction.ROLL_CALL):
            if self.roll_reason is None:
                raise ValueError(f"{self.action.value} proposals must include roll_reason")
            if self.closing_position_premium is None:
                raise ValueError(f"{self.action.value} proposals must include closing_position_premium")

        # Right must match the put/call action family
        if self.action in (WheelAction.SELL_PUT, WheelAction.BUY_PUT, WheelAction.ROLL_PUT):
            if self.right != OptionRight.PUT:
                raise ValueError(
                    f"{self.action.value} requires right={OptionRight.PUT.value}"
                )
        elif self.action in (WheelAction.SELL_CALL, WheelAction.BUY_CALL, WheelAction.ROLL_CALL):
            if self.right != OptionRight.CALL:
                raise ValueError(
                    f"{self.action.value} requires right={OptionRight.CALL.value}"
                )

        return self


def render_options_proposal(proposal: OptionsProposal) -> str:
    """Render an OptionsProposal to the markdown shape memory/logs consume."""
    parts = [
        f"**Symbol**: {proposal.symbol}",
        f"**Cycle Stage**: {proposal.cycle_stage.value}",
        f"**Action**: {proposal.action.value}",
    ]
    if proposal.action == WheelAction.NO_OP:
        parts.append(f"**No-Op Reason**: {proposal.no_op_reason.value}")
    else:
        parts.extend([
            f"**Right / Strike / Expiry**: {proposal.right.value} {proposal.strike} {proposal.expiry}",
            f"**Contracts**: {proposal.contracts}",
        ])
        if proposal.delta is not None:
            parts.append(f"**Delta**: {proposal.delta:+.3f}")
        if proposal.expected_premium is not None:
            parts.append(f"**Expected Premium**: {proposal.expected_premium}")
        if proposal.dte is not None:
            parts.append(f"**DTE**: {proposal.dte}")
        if proposal.roll_reason is not None:
            parts.append(f"**Roll Reason**: {proposal.roll_reason.value}")
        if proposal.closing_position_premium is not None:
            parts.append(f"**Closing Leg Premium**: {proposal.closing_position_premium}")
    parts.extend(["", f"**Rationale**: {proposal.rationale}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio-level final decision
# ---------------------------------------------------------------------------


class RiskGateResult(BaseModel):
    """Outcome of a single risk gate check, for audit trail."""

    name: str = Field(description="Gate identifier, e.g. 'earnings_blackout', 'buying_power'")
    passed: bool
    detail: Optional[str] = None


class WheelDecision(BaseModel):
    """Final decision the OptionsAgents graph hands to the executor.

    Wraps the selected proposal with the full audit trail: which gates were
    evaluated, what the regime was, and the synthesizer's executive summary.
    """

    selected_proposal: OptionsProposal
    competing_proposals: List[OptionsProposal] = Field(
        default_factory=list,
        description="Other proposals that were considered but not selected this run",
    )
    risk_gates: List[RiskGateResult] = Field(default_factory=list)
    executive_summary: str = Field(
        description="Two to four sentences synthesizing why this proposal was selected over the alternatives.",
    )
    regime_at_decision: Optional[str] = Field(default=None)


def render_wheel_decision(decision: WheelDecision) -> str:
    """Render a WheelDecision to markdown for memory log + Telegram alerts."""
    parts = [
        "## Wheel Decision",
        "",
        render_options_proposal(decision.selected_proposal),
        "",
        f"**Executive Summary**: {decision.executive_summary}",
    ]
    if decision.regime_at_decision:
        parts.append(f"**Regime**: {decision.regime_at_decision}")
    if decision.risk_gates:
        parts.extend(["", "**Risk Gates**:"])
        for g in decision.risk_gates:
            status = "✓" if g.passed else "✗"
            line = f"- {status} {g.name}"
            if g.detail:
                line += f" — {g.detail}"
            parts.append(line)
    if decision.competing_proposals:
        parts.extend(["", f"**Competing Proposals Considered**: {len(decision.competing_proposals)}"])
        for p in decision.competing_proposals:
            parts.append(f"- {p.symbol}: {p.action.value}")
    final_action = decision.selected_proposal.action.value
    parts.extend(["", f"FINAL WHEEL ACTION: **{final_action}**"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Cycle outcome — the reflection loop record
# ---------------------------------------------------------------------------


class CycleOutcomeStatus(str, Enum):
    """How a wheel-cycle leg resolved."""

    # Plain-English values written by the reflection agent (Stage 4).
    EXPIRED_WORTHLESS = "expired_worthless"        # Best case for STO leg
    CLOSED_FOR_PROFIT = "closed_for_profit"        # BTC at target
    CLOSED_DEFENSIVELY = "closed_defensively"      # BTC at loss to limit damage
    ASSIGNED = "put_assigned_shares_acquired"      # Put assigned → shares acquired
    CALLED_AWAY = "call_assigned_shares_delivered" # Call assigned → shares delivered
    ROLLED = "rolled_to_new_leg"                   # Position rolled to new leg


class CycleOutcome(BaseModel):
    """Realized outcome of a single wheel-cycle leg, for memory/reflection.

    Mirrors TradingAgents' pending→resolved pattern but on wheel-relevant
    dimensions: premium captured, DTE held, assignment outcome, realized
    return *as a fraction of capital at risk* (not vs SPY — wheel cycles
    have their own benchmark logic).
    """

    decision_id: str = Field(description="Stable ID linking back to the originating WheelDecision")
    symbol: str
    action_taken: WheelAction
    open_date: date
    close_date: date
    status: CycleOutcomeStatus
    premium_collected: float = Field(description="Net premium received over the leg (positive)")
    premium_paid_to_close: float = Field(default=0.0, description="Premium paid to close, if any")
    capital_at_risk: float = Field(description="Strike * 100 * contracts for puts; share value for calls")
    realized_return_pct: float = Field(description="Net P&L / capital_at_risk")
    days_held: int
    regime_at_open: Optional[str] = None
    setup_signature: Optional[str] = Field(
        default=None,
        description="Bucketed signature for memory keying, e.g. 'put_delta20_dte30_chop'",
    )


def render_cycle_outcome(outcome: CycleOutcome) -> str:
    """Render a CycleOutcome to markdown for the resolved memory log."""
    parts = [
        f"**Symbol / Action**: {outcome.symbol} / {outcome.action_taken.value}",
        f"**Open → Close**: {outcome.open_date} → {outcome.close_date} ({outcome.days_held}d)",
        f"**Status**: {outcome.status.value}",
        f"**Premium Collected**: {outcome.premium_collected}",
    ]
    if outcome.premium_paid_to_close:
        parts.append(f"**Premium Paid To Close**: {outcome.premium_paid_to_close}")
    parts.extend([
        f"**Capital At Risk**: {outcome.capital_at_risk}",
        f"**Realized Return**: {outcome.realized_return_pct:+.4f}",
    ])
    if outcome.regime_at_open:
        parts.append(f"**Regime At Open**: {outcome.regime_at_open}")
    if outcome.setup_signature:
        parts.append(f"**Setup Signature**: {outcome.setup_signature}")
    return "\n".join(parts)
