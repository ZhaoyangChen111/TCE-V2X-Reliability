from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HotspotWeightParams:
    cross_mult: float = 1.04
    turn_mult: float = 1.06
    low_speed_thresh_mps: float = 3.0
    low_speed_bonus: float = 0.02
    queue_speed_thresh_mps: float = 1.0
    queue_bonus: float = 0.02


def hotspot_multipliers(
    road_tag: str,
    tx_speed_mps: float | None,
    params: HotspotWeightParams,
) -> tuple[float, float]:
    """
    Returns (collision_multiplier, delay_multiplier).
    """
    tag = (road_tag or "").upper()
    col_mult = 1.0
    delay_mult = 1.0

    if tag.startswith("CROSS_"):
        col_mult *= float(params.cross_mult)
        delay_mult *= float(params.cross_mult)
    elif tag.startswith("TURN_"):
        col_mult *= float(params.turn_mult)
        delay_mult *= float(params.turn_mult)

    if tx_speed_mps is not None:
        v = float(tx_speed_mps)
        if v <= float(params.low_speed_thresh_mps):
            col_mult *= (1.0 + float(params.low_speed_bonus))
            delay_mult *= (1.0 + float(params.low_speed_bonus))
        if v <= float(params.queue_speed_thresh_mps):
            col_mult *= (1.0 + float(params.queue_bonus))
            delay_mult *= (1.0 + float(params.queue_bonus))

    return float(col_mult), float(delay_mult)