import pytest

from stockscan.sizing import FeeModel, max_loss_eur, size_position, trade_costs

NO_FEES = FeeModel(commission_min_eur=0.0, commission_pct=0.0, fx_fee_pct=0.0)
REVOLUT_USD = FeeModel(commission_min_eur=1.0, commission_pct=0.25, fx_fee_pct=0.0)
REVOLUT_EUR = FeeModel(commission_min_eur=1.0, commission_pct=0.25, fx_fee_pct=1.0)


class TestFeeModel:
    def test_commission_is_the_larger_of_minimum_and_percentage(self):
        assert REVOLUT_USD.commission_eur(100.0) == 1.0  # 0.25% = 0.25 < 1 minimum
        assert REVOLUT_USD.commission_eur(1000.0) == 2.5  # 0.25% = 2.50 > 1 minimum
        assert REVOLUT_USD.commission_eur(0.0) == 0.0  # no trade, no fee

    def test_fx_fee_is_a_percentage_of_notional(self):
        assert REVOLUT_EUR.fx_fee_eur(500.0) == 5.0
        assert REVOLUT_USD.fx_fee_eur(500.0) == 0.0


def test_trade_costs_and_max_loss_by_hand():
    # 10 shares, entry $110, stop $99, EUR/USD 1.10 -> entry EUR 1000, exit EUR 900.
    costs = trade_costs(10, 110.0, 99.0, 1.10, REVOLUT_EUR)
    assert costs.position_eur == pytest.approx(1000.0)
    assert costs.entry_fees_eur == pytest.approx(2.5 + 10.0)  # 0.25% commission + 1% FX
    assert costs.exit_fees_eur == pytest.approx(2.25 + 9.0)
    assert costs.total_cost_eur == pytest.approx(1012.5)
    # price loss 10 * $11 / 1.10 = EUR 100, plus EUR 23.75 of fees
    assert max_loss_eur(10, 110.0, 99.0, 1.10, REVOLUT_EUR) == pytest.approx(123.75)
    assert max_loss_eur(0, 110.0, 99.0, 1.10, REVOLUT_EUR) == 0.0


class TestWholeShares:
    def test_textbook_case_without_fees(self):
        # EUR 10,000 pot, 1% risk = EUR 100 = $110; $2 risk per share -> 55 shares.
        result = size_position(10_000, entry=50.0, stop=48.0, eurusd=1.10, fees=NO_FEES)
        assert result.shares == 55
        assert result.fractional is False
        assert result.risk_budget_eur == pytest.approx(100.0)
        assert result.max_loss_usd == pytest.approx(110.0)
        assert result.max_loss_eur == pytest.approx(100.0)
        assert result.risk_pct_of_pot == pytest.approx(1.0)
        assert result.position_usd == pytest.approx(2750.0)
        assert result.total_cost_eur == pytest.approx(2500.0)
        assert result.limited_by == "risk"
        assert result.warnings == ()

    def test_fees_reduce_the_share_count(self):
        without = size_position(10_000, 50.0, 48.0, 1.10, NO_FEES)
        with_fees = size_position(10_000, 50.0, 48.0, 1.10, REVOLUT_EUR)
        assert with_fees.shares < without.shares
        assert with_fees.max_loss_eur <= with_fees.risk_budget_eur
        # One more share would break the budget: the answer is the largest that fits.
        assert max_loss_eur(with_fees.shares + 1, 50.0, 48.0, 1.10, REVOLUT_EUR) > 100.0

    def test_default_risk_is_one_percent(self):
        result = size_position(5_000, 20.0, 19.0, 1.0, NO_FEES)
        assert result.risk_pct_requested == 1.0
        assert result.shares == 50

    def test_shares_are_rounded_down_never_up(self):
        # Budget $110, $3 per share -> 36.67 -> 36.
        result = size_position(10_000, 50.0, 47.0, 1.10, NO_FEES)
        assert result.shares == 36
        assert result.risk_pct_of_pot < 1.0

    def test_no_leverage_position_is_capped_by_the_pot(self):
        # Tight stop: the risk budget alone would allow 1000 shares = $100k on a $1.1k pot.
        result = size_position(1_000, entry=100.0, stop=99.99, eurusd=1.10, fees=NO_FEES)
        assert result.shares == 11  # 11 * $100 = $1100 = EUR 1000
        assert result.total_cost_eur <= 1_000 + 1e-6
        assert result.limited_by == "pot"
        assert any("capped by the pot" in warning for warning in result.warnings)

    def test_small_pot_cannot_afford_one_share_within_budget(self):
        # EUR 500 pot -> EUR 5 budget; one share risks $20.
        result = size_position(500, entry=400.0, stop=380.0, eurusd=1.10, fees=REVOLUT_USD)
        assert result.shares == 0
        assert result.max_loss_eur == 0.0
        assert result.total_cost_eur == 0.0
        assert result.limited_by == "none"
        assert any("No position fits" in warning for warning in result.warnings)


class TestFractionalShares:
    def test_fractional_uses_the_budget_more_fully_than_whole_shares(self):
        whole = size_position(10_000, 50.0, 47.0, 1.10, NO_FEES)
        fractional = size_position(10_000, 50.0, 47.0, 1.10, NO_FEES, fractional=True)
        assert fractional.fractional is True
        assert fractional.shares == pytest.approx(36.6666, abs=1e-4)
        assert whole.shares <= fractional.shares < whole.shares + 1
        assert fractional.max_loss_eur <= fractional.risk_budget_eur + 1e-9

    def test_fractional_makes_a_small_pot_tradable_when_fees_allow(self):
        whole = size_position(500, 400.0, 380.0, 1.10, NO_FEES)
        fractional = size_position(500, 400.0, 380.0, 1.10, NO_FEES, fractional=True)
        assert whole.shares == 0
        # EUR 5 budget = $5.50; $20 risk per share -> 0.275 shares.
        assert fractional.shares == pytest.approx(0.275)

    def test_minimum_fees_can_consume_most_of_a_small_budget(self):
        # EUR 5 budget, EUR 2 of minimum commissions: only EUR 3 is left for price risk.
        result = size_position(500, 400.0, 380.0, 1.10, REVOLUT_USD, fractional=True)
        assert result.shares == pytest.approx(3.0 * 1.10 / 20.0, abs=1e-4)
        assert result.fees_eur == pytest.approx(2.0)
        assert any("expensive for its size" in warning for warning in result.warnings)

    def test_fees_larger_than_the_budget_leave_nothing(self):
        result = size_position(150, 400.0, 380.0, 1.10, REVOLUT_USD, fractional=True)
        assert result.shares == 0  # EUR 1.50 budget < EUR 2 of minimum fees

    def test_decimals_are_configurable(self):
        result = size_position(
            10_000, 50.0, 47.0, 1.10, NO_FEES, fractional=True, fractional_decimals=1
        )
        assert result.shares == pytest.approx(36.6)


class TestWarningsAndValidation:
    def test_warns_above_two_percent_risk(self):
        risky = size_position(10_000, 50.0, 48.0, 1.10, NO_FEES, risk_pct=2.5)
        assert any("above the 2% limit" in warning for warning in risky.warnings)
        assert risky.shares == 137  # EUR 250 = $275 at $2 per share, rounded down
        assert risky.risk_pct_of_pot == pytest.approx(2.49, abs=0.01)

    def test_exactly_two_percent_does_not_warn(self):
        assert size_position(10_000, 50.0, 48.0, 1.10, NO_FEES, risk_pct=2.0).warnings == ()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"pot_eur": 0}, "pot"),
            ({"entry": -1.0}, "positive prices"),
            ({"stop": 50.0}, "below entry"),
            ({"stop": 55.0}, "below entry"),
            ({"eurusd": 0}, "EUR/USD"),
            ({"risk_pct": 0}, "risk %"),
            ({"risk_pct": 150}, "risk %"),
        ],
    )
    def test_invalid_inputs_are_rejected(self, kwargs, message):
        arguments = {"pot_eur": 1000, "entry": 50.0, "stop": 48.0, "eurusd": 1.1, "fees": NO_FEES}
        arguments.update(kwargs)
        with pytest.raises(ValueError, match=message):
            size_position(**arguments)

    @pytest.mark.parametrize("fractional", [False, True])
    @pytest.mark.parametrize("fees", [NO_FEES, REVOLUT_USD, REVOLUT_EUR])
    @pytest.mark.parametrize(
        ("pot", "entry", "stop"),
        [(300, 12.5, 11.9), (2_000, 172.3, 165.0), (50_000, 640.0, 628.5), (800, 3.2, 3.05)],
    )
    def test_result_never_exceeds_budget_or_pot(self, pot, entry, stop, fees, fractional):
        result = size_position(pot, entry, stop, 1.08, fees, fractional=fractional)
        assert result.max_loss_eur <= result.risk_budget_eur + 1e-6
        assert result.total_cost_eur <= pot + 1e-6
        assert result.risk_pct_of_pot <= 1.0 + 1e-6
