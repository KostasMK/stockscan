"""Position sizing for long share positions, with fees and EUR/USD conversion.

Answers one question: "given my pot and the risk I accept, how many shares may I buy so
that being stopped out costs no more than that?" It only does arithmetic; it never
places or prepares an order.

Model (all fees configurable):

* commission per side = max(commission_min_eur, commission_pct% of the notional)
* FX fee = fx_fee_pct% of the notional, charged on the way in and on the way out
* max loss = shares * (entry - stop) + entry and exit commissions + both FX fees,
  assuming the stop fills exactly at the stop price (a gap through the stop loses more)
* no leverage: the position plus entry costs can never exceed the pot
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class FeeModel:
    commission_min_eur: float = 1.0
    commission_pct: float = 0.25
    fx_fee_pct: float = 0.0

    def commission_eur(self, notional_eur: float) -> float:
        if notional_eur <= 0:
            return 0.0
        return max(self.commission_min_eur, notional_eur * self.commission_pct / 100.0)

    def fx_fee_eur(self, notional_eur: float) -> float:
        return max(0.0, notional_eur) * self.fx_fee_pct / 100.0


@dataclass(frozen=True)
class TradeCosts:
    position_eur: float
    """Shares times entry price, in EUR."""
    entry_fees_eur: float
    exit_fees_eur: float
    """Fees when selling at the stop price."""

    @property
    def fees_eur(self) -> float:
        return self.entry_fees_eur + self.exit_fees_eur

    @property
    def total_cost_eur(self) -> float:
        """Cash needed to open the position."""
        return self.position_eur + self.entry_fees_eur


def trade_costs(
    shares: float, entry: float, stop: float, eurusd: float, fees: FeeModel
) -> TradeCosts:
    """Cash and fees for buying ``shares`` at ``entry`` and selling them at ``stop``."""
    entry_notional = shares * entry / eurusd
    exit_notional = shares * stop / eurusd
    return TradeCosts(
        position_eur=entry_notional,
        entry_fees_eur=fees.commission_eur(entry_notional) + fees.fx_fee_eur(entry_notional),
        exit_fees_eur=fees.commission_eur(exit_notional) + fees.fx_fee_eur(exit_notional),
    )


def max_loss_eur(shares: float, entry: float, stop: float, eurusd: float, fees: FeeModel) -> float:
    """Loss in EUR if stopped out exactly at ``stop``, fees included."""
    if shares <= 0:
        return 0.0
    costs = trade_costs(shares, entry, stop, eurusd, fees)
    return shares * (entry - stop) / eurusd + costs.fees_eur


@dataclass(frozen=True)
class PositionSize:
    shares: float
    fractional: bool
    pot_eur: float
    entry: float
    stop: float
    eurusd: float
    risk_pct_requested: float
    risk_budget_eur: float
    position_eur: float
    position_usd: float
    fees_eur: float
    total_cost_eur: float
    total_cost_usd: float
    max_loss_eur: float
    max_loss_usd: float
    risk_pct_of_pot: float
    limited_by: str
    """"risk" (the risk budget binds), "pot" (cash binds) or "none" (zero shares)."""
    warnings: tuple[str, ...]


def _largest_feasible(upper: int, feasible: Callable[[int], bool]) -> int:
    """Largest n in [0, upper] with feasible(n), for a predicate that is true up to a point."""
    low, high = 0, upper
    while low < high:
        middle = (low + high + 1) // 2
        if feasible(middle):
            low = middle
        else:
            high = middle - 1
    return low


def size_position(
    pot_eur: float,
    entry: float,
    stop: float,
    eurusd: float,
    fees: FeeModel,
    risk_pct: float = 1.0,
    fractional: bool = False,
    fractional_decimals: int = 4,
    max_risk_pct_warning: float = 2.0,
) -> PositionSize:
    """Largest long position whose worst case (stop hit, fees paid) fits the risk budget.

    ``entry`` and ``stop`` are USD prices, ``eurusd`` is USD per 1 EUR. With
    ``fractional=False`` the result is whole shares (what a limit order needs); with
    ``fractional=True`` shares are rounded down to ``fractional_decimals`` places.
    """
    if pot_eur <= 0:
        raise ValueError("pot must be positive")
    if entry <= 0 or stop <= 0:
        raise ValueError("entry and stop must be positive prices")
    if stop >= entry:
        raise ValueError(
            "stop must be below entry: this helper sizes long share positions only (no shorting)"
        )
    if eurusd <= 0:
        raise ValueError("EUR/USD rate must be positive")
    if not 0 < risk_pct <= 100:
        raise ValueError("risk % must be between 0 and 100")

    budget = pot_eur * risk_pct / 100.0
    units_per_share = 10**fractional_decimals if fractional else 1
    tolerance = 1e-9

    def loss_ok(units: int) -> bool:
        return (
            max_loss_eur(units / units_per_share, entry, stop, eurusd, fees) <= budget + tolerance
        )

    def cash_ok(units: int) -> bool:
        costs = trade_costs(units / units_per_share, entry, stop, eurusd, fees)
        return costs.total_cost_eur <= pot_eur + tolerance

    # Ignoring fees gives an upper bound; fees only ever shrink the position.
    upper_shares = min(budget * eurusd / (entry - stop), pot_eur * eurusd / entry)
    upper = int(upper_shares * units_per_share)
    units = _largest_feasible(upper, lambda n: n == 0 or (loss_ok(n) and cash_ok(n)))
    shares = units / units_per_share

    costs = trade_costs(shares, entry, stop, eurusd, fees)
    loss = max_loss_eur(shares, entry, stop, eurusd, fees)
    if units == 0:
        limited_by = "none"
    elif not loss_ok(units + 1):
        limited_by = "risk"
    else:
        limited_by = "pot"

    warnings: list[str] = []
    if risk_pct > max_risk_pct_warning:
        warnings.append(
            f"Risking {risk_pct:g}% of the pot on one trade is above the "
            f"{max_risk_pct_warning:g}% limit you set for yourself."
        )
    if units == 0:
        smallest = 1 / units_per_share
        smallest_loss = max_loss_eur(smallest, entry, stop, eurusd, fees)
        kind = "share" if not fractional else f"{smallest:g} of a share"
        warnings.append(
            f"No position fits: even one {kind} would lose EUR {smallest_loss:.2f} at the stop "
            f"(fees included), more than the EUR {budget:.2f} risk budget or the pot allows. "
            "Skip the trade, or rethink the stop; do not raise the risk to make it fit."
        )
    else:
        if limited_by == "pot":
            warnings.append(
                "Position is capped by the pot (no leverage), so the actual risk is below "
                f"the requested {risk_pct:g}%."
            )
        if loss > 0 and costs.fees_eur / loss > 0.25:
            warnings.append(
                f"Fees (EUR {costs.fees_eur:.2f}) are {costs.fees_eur / loss:.0%} of the maximum "
                "loss: this trade is expensive for its size."
            )

    return PositionSize(
        shares=shares,
        fractional=fractional,
        pot_eur=pot_eur,
        entry=entry,
        stop=stop,
        eurusd=eurusd,
        risk_pct_requested=risk_pct,
        risk_budget_eur=budget,
        position_eur=costs.position_eur,
        position_usd=costs.position_eur * eurusd,
        fees_eur=costs.fees_eur,
        total_cost_eur=costs.total_cost_eur,
        total_cost_usd=costs.total_cost_eur * eurusd,
        max_loss_eur=loss,
        max_loss_usd=loss * eurusd,
        risk_pct_of_pot=loss / pot_eur * 100.0,
        limited_by=limited_by,
        warnings=tuple(warnings),
    )
