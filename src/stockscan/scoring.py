"""Transparent ranking score.

    score = 100 * sum(weight_i * component_i) / sum(weights)

Each component is a metric squashed to [0, 1] by a cap from the config file. The score
is direction-neutral ("how unusual was today?"); direction is reported separately.
Every row keeps its per-component contribution (``c_*`` columns, in score points, adding
up to the score) and a one-line reason, so a ranking can always be explained.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

from stockscan.config import ScoringCaps, ScoringConfig

COMPONENTS = ["rel_volume", "gap", "move", "extreme", "stretch"]
CONTRIBUTION_COLUMNS = [f"c_{name}" for name in COMPONENTS]
COMPONENT_LABELS = {
    "rel_volume": "rel. volume",
    "gap": "gap",
    "move": "day move",
    "extreme": "52w extreme",
    "stretch": "SMA20 stretch",
}
MAX_REASON_PARTS = 3
MIN_REASON_POINTS = 2.0  # contributions smaller than this are not worth mentioning
AT_EXTREME_PCT = 0.5  # within this distance we just say "at 52w high/low"


def component_values(metrics: pd.DataFrame, caps: ScoringCaps) -> pd.DataFrame:
    """Raw components in [0, 1], one column per entry of ``COMPONENTS``.

    A metric that is NaN (not enough history) contributes 0 rather than poisoning the score.
    """
    nearest_extreme = pd.concat(
        [metrics["dist_52w_high_pct"].abs(), metrics["dist_52w_low_pct"].abs()], axis=1
    ).min(axis=1)
    raw = pd.DataFrame(
        {
            "rel_volume": (metrics["rel_volume"] - 1.0) / (caps.rel_volume - 1.0),
            "gap": metrics["gap_pct"].abs() / caps.gap_pct,
            "move": metrics["move_atr"].abs() / caps.move_atr,
            "extreme": 1.0 - nearest_extreme / caps.extreme_pct,
            "stretch": metrics["close_vs_sma20_pct"].abs() / caps.stretch_pct,
        },
        index=metrics.index,
    )
    return raw[COMPONENTS].clip(lower=0.0, upper=1.0).fillna(0.0)


def direction_of(day_change_pct: float) -> str:
    if isinstance(day_change_pct, float) and math.isnan(day_change_pct):
        return "flat"
    if day_change_pct > 0:
        return "up"
    if day_change_pct < 0:
        return "down"
    return "flat"


def _describe(component: str, row: Mapping[str, Any]) -> str:
    if component == "rel_volume":
        return f"volume {row['rel_volume']:.1f}x average"
    if component == "gap":
        return f"gap {row['gap_pct']:+.1f}%"
    if component == "move":
        return f"{row['day_change_pct']:+.1f}% on the day ({abs(row['move_atr']):.1f} ATR)"
    if component == "extreme":
        below_high = abs(row["dist_52w_high_pct"])
        above_low = abs(row["dist_52w_low_pct"])
        if not above_low < below_high:  # also true when the low distance is NaN
            return (
                "at 52w high"
                if below_high <= AT_EXTREME_PCT
                else f"{below_high:.1f}% below 52w high"
            )
        return "at 52w low" if above_low <= AT_EXTREME_PCT else f"{above_low:.1f}% above 52w low"
    if component == "stretch":
        return f"{row['close_vs_sma20_pct']:+.1f}% vs 20d SMA"
    raise ValueError(f"unknown component: {component}")


def build_reason(row: Mapping[str, Any]) -> str:
    """One line naming the biggest contributors to the score, largest first."""
    ranked = sorted(COMPONENTS, key=lambda name: row[f"c_{name}"], reverse=True)
    parts = [
        _describe(name, row)
        for name in ranked[:MAX_REASON_PARTS]
        if row[f"c_{name}"] >= MIN_REASON_POINTS
    ]
    return "; ".join(parts) if parts else "nothing unusual"


def score_universe(metrics: pd.DataFrame, config: ScoringConfig) -> pd.DataFrame:
    """Add contributions, score, direction, reason and rank to one day's metric rows."""
    weights = config.weights.model_dump()
    total_weight = sum(weights.values())
    components = component_values(metrics, config.caps)

    scored = metrics.copy()
    for name in COMPONENTS:
        scored[f"c_{name}"] = 100.0 * weights[name] / total_weight * components[name]
    scored["score"] = scored[CONTRIBUTION_COLUMNS].sum(axis=1)
    scored["direction"] = [direction_of(float(value)) for value in scored["day_change_pct"]]
    scored["reason"] = [build_reason(row) for row in scored.to_dict("records")]

    scored = scored.sort_values(["score", "ticker"], ascending=[False, True], kind="stable")
    scored["rank"] = range(1, len(scored) + 1)
    return scored.reset_index(drop=True)
