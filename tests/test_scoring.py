import numpy as np
import pandas as pd
import pytest

from stockscan.config import ScoringCaps, ScoringConfig, ScoringWeights
from stockscan.scoring import (
    COMPONENTS,
    CONTRIBUTION_COLUMNS,
    build_reason,
    component_values,
    direction_of,
    score_universe,
)


def metric_rows(**overrides: list[float]) -> pd.DataFrame:
    """Three quiet tickers; tests override single metrics to make one of them 'hot'."""
    base = {
        "ticker": ["AAA", "BBB", "CCC"],
        "rel_volume": [1.0, 1.0, 1.0],
        "gap_pct": [0.0, 0.0, 0.0],
        "day_change_pct": [0.0, 0.0, 0.0],
        "move_atr": [0.0, 0.0, 0.0],
        "dist_52w_high_pct": [-30.0, -30.0, -30.0],
        "dist_52w_low_pct": [30.0, 30.0, 30.0],
        "close_vs_sma20_pct": [0.0, 0.0, 0.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


CAPS = ScoringCaps()  # rel_volume 5, gap 5%, move 3 ATR, extreme 10%, stretch 10%


class TestComponentValues:
    def test_quiet_stock_scores_zero_everywhere(self):
        assert (component_values(metric_rows(), CAPS) == 0).all().all()

    def test_each_component_scales_linearly_and_saturates_at_its_cap(self):
        frame = metric_rows(
            rel_volume=[3.0, 5.0, 9.0],  # (3-1)/(5-1) = 0.5, 1.0, clipped 1.0
            gap_pct=[2.5, -5.0, 12.0],  # 0.5, 1.0 (sign ignored), clipped
            move_atr=[1.5, -3.0, 6.0],
            close_vs_sma20_pct=[5.0, -10.0, 25.0],
        )
        values = component_values(frame, CAPS)
        for name in ["rel_volume", "gap", "move", "stretch"]:
            assert values[name].tolist() == pytest.approx([0.5, 1.0, 1.0]), name

    def test_below_average_volume_does_not_go_negative(self):
        values = component_values(metric_rows(rel_volume=[0.2, 0.9, 1.0]), CAPS)
        assert values["rel_volume"].tolist() == [0.0, 0.0, 0.0]

    def test_extreme_rewards_closeness_to_either_end_of_the_range(self):
        frame = metric_rows(
            dist_52w_high_pct=[0.0, -50.0, -5.0],
            dist_52w_low_pct=[80.0, 0.0, 40.0],
        )
        # at the high -> 1, at the low -> 1, 5% below the high -> 0.5
        assert component_values(frame, CAPS)["extreme"].tolist() == pytest.approx([1.0, 1.0, 0.5])

    def test_missing_metrics_contribute_zero(self):
        frame = metric_rows(
            rel_volume=[np.nan, 5.0, 1.0],
            dist_52w_high_pct=[np.nan, -30.0, -30.0],
            dist_52w_low_pct=[np.nan, 30.0, 30.0],
        )
        values = component_values(frame, CAPS)
        assert values.loc[0].tolist() == [0.0] * len(COMPONENTS)
        assert values.notna().all().all()


class TestScoreUniverse:
    def test_score_is_the_weighted_sum_and_contributions_add_up(self):
        frame = metric_rows(rel_volume=[3.0, 1.0, 1.0], gap_pct=[5.0, 0.0, 0.0])
        scored = score_universe(frame, ScoringConfig())
        top = scored.iloc[0]
        assert top["ticker"] == "AAA"
        # default weights: rel_volume 0.30 * 0.5 + gap 0.20 * 1.0 = 0.35 -> 35 points
        assert top["c_rel_volume"] == pytest.approx(15.0)
        assert top["c_gap"] == pytest.approx(20.0)
        assert top["score"] == pytest.approx(35.0)
        assert scored[CONTRIBUTION_COLUMNS].sum(axis=1).tolist() == pytest.approx(
            scored["score"].tolist()
        )

    def test_everything_maxed_out_scores_100(self):
        frame = metric_rows(
            rel_volume=[9.0, 1.0, 1.0],
            gap_pct=[9.0, 0.0, 0.0],
            move_atr=[9.0, 0.0, 0.0],
            dist_52w_high_pct=[0.0, -30.0, -30.0],
            close_vs_sma20_pct=[30.0, 0.0, 0.0],
        )
        assert score_universe(frame, ScoringConfig())["score"].iloc[0] == pytest.approx(100.0)

    def test_weights_come_from_config_and_are_normalised(self):
        frame = metric_rows(rel_volume=[5.0, 1.0, 1.0], gap_pct=[0.0, 5.0, 0.0])
        volume_only = ScoringConfig(
            weights=ScoringWeights(rel_volume=7, gap=0, move=0, extreme=0, stretch=0)
        )
        scored = score_universe(frame, volume_only).set_index("ticker")
        assert scored.loc["AAA", "score"] == pytest.approx(100.0)  # 7/7, not 700
        assert scored.loc["BBB", "score"] == pytest.approx(0.0)

        gap_heavy = ScoringConfig(
            weights=ScoringWeights(rel_volume=1, gap=3, move=0, extreme=0, stretch=0)
        )
        assert score_universe(frame, gap_heavy).iloc[0]["ticker"] == "BBB"

    def test_score_is_direction_neutral_but_direction_is_labelled(self):
        frame = metric_rows(
            gap_pct=[4.0, -4.0, 0.0],
            day_change_pct=[6.0, -6.0, 0.0],
            move_atr=[2.0, -2.0, 0.0],
        )
        scored = score_universe(frame, ScoringConfig()).set_index("ticker")
        assert scored.loc["AAA", "score"] == pytest.approx(scored.loc["BBB", "score"])
        assert scored["direction"].to_dict() == {"AAA": "up", "BBB": "down", "CCC": "flat"}

    def test_rank_orders_by_score_with_ticker_as_tie_break(self):
        frame = metric_rows(rel_volume=[1.0, 5.0, 5.0])
        scored = score_universe(frame, ScoringConfig())
        assert scored["ticker"].tolist() == ["BBB", "CCC", "AAA"]
        assert scored["rank"].tolist() == [1, 2, 3]

    def test_input_frame_is_not_modified(self):
        frame = metric_rows()
        score_universe(frame, ScoringConfig())
        assert "score" not in frame.columns


def test_direction_of():
    assert [direction_of(x) for x in (1.2, -0.1, 0.0, float("nan"))] == [
        "up",
        "down",
        "flat",
        "flat",
    ]


class TestBuildReason:
    def row(self, **overrides: float) -> dict:
        row = {
            "rel_volume": 3.2,
            "gap_pct": 4.1,
            "day_change_pct": 6.0,
            "move_atr": 2.0,
            "dist_52w_high_pct": -0.2,
            "dist_52w_low_pct": 80.0,
            "close_vs_sma20_pct": 7.5,
            "c_rel_volume": 16.5,
            "c_gap": 16.4,
            "c_move": 16.7,
            "c_extreme": 14.7,
            "c_stretch": 7.5,
        }
        row.update(overrides)
        return row

    def test_names_the_three_biggest_contributors_in_order(self):
        assert build_reason(self.row()) == (
            "+6.0% on the day (2.0 ATR); volume 3.2x average; gap +4.1%"
        )

    def test_small_contributions_are_left_out(self):
        reason = build_reason(self.row(c_gap=1.0, c_move=0.0, c_extreme=0.5, c_stretch=0.0))
        assert reason == "volume 3.2x average"

    def test_extreme_wording(self):
        only_extreme = {"c_rel_volume": 0, "c_gap": 0, "c_move": 0, "c_stretch": 0}
        assert build_reason(self.row(**only_extreme)) == "at 52w high"
        assert (
            build_reason(self.row(**only_extreme, dist_52w_high_pct=-3.0)) == "3.0% below 52w high"
        )
        near_low = {"dist_52w_high_pct": -60.0, "dist_52w_low_pct": 2.5}
        assert build_reason(self.row(**only_extreme, **near_low)) == "2.5% above 52w low"
        at_low = {"dist_52w_high_pct": -60.0, "dist_52w_low_pct": 0.0}
        assert build_reason(self.row(**only_extreme, **at_low)) == "at 52w low"

    def test_quiet_stock(self):
        quiet = dict.fromkeys(CONTRIBUTION_COLUMNS, 0.0)
        assert build_reason(self.row(**quiet)) == "nothing unusual"
