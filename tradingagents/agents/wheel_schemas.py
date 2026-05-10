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

    PUT = "P"
    CALL = "C"


class WheelAction(str, Enum):
    """All actions an OptionsAgents proposal can take.

    The wheel has four mechanical legs plus an explicit no-op. Rolls are
    represented as compound actions because the executor needs to know it's
    a single net-credit transaction, not two independent legs.
    """

    SELL_PUT = "STO_PUT"               # Open: sell cash-secured put
    BUY_PUT = "BTC_PUT"                # Close: buy back short put (profit-take or defensive)
    SELL_CALL = "STO_CALL"             # Open: sell covered call against assigned shares
    BUY_CALL = "BTC_CALL"              # Close: buy back short call (profit-take or defensive)
    ROLL_PUT = "ROLL_PUT"              # Compound: BTC current put + STO new put
    ROLL_CALL = "ROLL_CALL"            # Compound: BTC current call + STO new call
    NO_OP = "NO_OP"                    # Explicit no-action with structured reason


class CycleStage(str, Enum):
    """Where this symbol currently sits in the wheel cycle."""

    CASH = "CASH"                      # No position; eligible to sell puts
    SHORT_PUT = "SHORT_PUT"            # Open short put; awaiting expiry or assignment
    ASSIGNED = "ASSIGNED"              # Shares assigned; eligible to sell calls
    SHORT_CALL = "SHORT_CALL"          # Open covered call; awaiting expiry or called-away


class NoOpReason(str, Enum):
    """Structured reasons for declining to act. Never silent."""

    EARNINGS_BLACKOUT = "EARNINGS_BLACKOUT"        # Earnings within configured window
    REGIME_UNFAVORABLE = "REGIME_UNFAVORABLE"      # regime.json blocks new opens
    CHAIN_UNAVAILABLE = "CHAIN_UNAVAILABLE"        # IBKR returned empty/stale chain
    NO_VIABLE_STRIKE = "NO_VIABLE_STRIKE"          # No strike met delta/premium criteria
    BUYING_POWER_INSUFFICIENT = "BUYING_POWER_INSUFFICIENT"
    CONCENTRATION_LIMIT = "CONCENTRATION_LIMIT"    # Position would exceed per-symbol cap
    POSITION_AT_PROFIT_TARGET = "POSITION_AT_PROFIT_TARGET"  # Hold for close, not roll
    DTE_OUTSIDE_BAND = "DTE_OUTSIDE_BAND"          # No expiries in target DTE window
    DUPLICATE_PROPOSAL = "DUPLICATE_PROPOSAL"      # Already have an open position matching
    OTHER = "OTHER"                                # Explained in free-text rationale


class RollReason(str, Enum):
    """Why a roll is being proposed (for memory/reflection tagging)."""

    DEFENSIVE_ITM = "DEFENSIVE_ITM"                # Strike breached, rolling out/down
    PROFIT_TAKE_AND_EXTEND = "PROFIT_TAKE_AND_EXTEND"  # ≥ profit target, harvest premium
    DTE_MANAGEMENT = "DTE_MANAGEMENT"              # Approaching expiry, no other reason


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
    cycle_stage: CycleStage = Field(description="Wheel stage this symbol is in right now")
    action: WheelAction = Field(description="Proposed action; NO_OP is valid and requires no_op_reason")

    # Trade economics — required when action is not NO_OP
    right: Optional[OptionRight] = Field(default=None, description="P or C; omit when action is NO_OP")
    strike: Optional[float] = Field(default=None, description="Strike price; omit when action is NO_OP")
    expiry: Optional[date] = Field(default=None, description="Contract expiry; omit when action is NO_OP")
    contracts: Optional[int] = Field(default=None, description="Number of contracts; positive integer")
    delta: Optional[float] = Field(default=None, description="Option delta at proposal time (signed)")
    expected_premium: Optional[float] = Field(default=None, description="Expected mid-price premium per share")
    dte: Optional[int] = Field(default=None, description="Days to expiry from today")

    # Roll-specific
    roll_reason: Optional[RollReason] = Field(default=None, description="Required when action is ROLL_*")
    closing_position_premium: Optional[float] = Field(
        default=None,
        description="Premium to pay to close the existing leg (for ROLL_* actions)",
    )

    # NO_OP path
    no_op_reason: Optional[NoOpReason] = Field(
        default=None,
        description="REQUIRED when action is NO_OP. Encodes why the agent declined to act.",
    )

    # Always required — natural language anchored in the analysis
    rationale: str = Field(
        description=(
            "Two to four sentences. For trade actions: justify strike, expiry, and "
            "sizing against the chain and the cycle stage. For NO_OP: explain "
            "the specific evidence behind no_op_reason."
        ),
    )

    @model_validator(mode="after")
    def _validate_action_consistency(self):
        """Enforce that schema fields are consistent with the chosen action.

        NO_OP must carry no_op_reason; trade actions must carry the legs;
        ROLL_* must carry roll_reason and closing_position_premium. This
        is the structural answer to the silent-failure regression — the
        proposal cannot validate as 'I won't do anything' without saying why.
        """
        if self.action == WheelAction.NO_OP:
            if self.no_op_reason is None:
                raise ValueError("NO_OP proposals must include no_op_reason")
            return self

        # All non-NO_OP actions require the core trade legs
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
                raise ValueError("ROLL_* proposals must include roll_reason")
            if self.closing_position_premium is None:
                raise ValueError("ROLL_* proposals must include closing_position_premium")

        # Right must match the put/call action family
        if self.action in (WheelAction.SELL_PUT, WheelAction.BUY_PUT, WheelAction.ROLL_PUT):
            if self.right != OptionRight.PUT:
                raise ValueError(f"{self.action.value} requires right=PUT")
        elif self.action in (WheelAction.SELL_CALL, WheelAction.BUY_CALL, WheelAction.ROLL_CALL):
            if self.right != OptionRight.CALL:
                raise ValueError(f"{self.action.value} requires right=CALL")

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

    EXPIRED_WORTHLESS = "EXPIRED_WORTHLESS"        # Best case for STO leg
    CLOSED_FOR_PROFIT = "CLOSED_FOR_PROFIT"        # BTC at target
    CLOSED_DEFENSIVELY = "CLOSED_DEFENSIVELY"      # BTC at loss to limit damage
    ASSIGNED = "ASSIGNED"                          # Put assigned → shares acquired
    CALLED_AWAY = "CALLED_AWAY"                    # Call assigned → shares delivered
    ROLLED = "ROLLED"                              # Position rolled to new leg


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
