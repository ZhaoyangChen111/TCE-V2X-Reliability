# modules/scenario_urbmaskplus.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field


@dataclass(frozen=True)
class UrbMaskBuildingsConfig:
    seed: int = 1
    n_blocks: int = 12
    x_margin_m: float = 10.0
    y_halfspan_mode: str = "cross_road"  # "cross_road" or "one_side"

    min_w_m: float = 18.0
    max_w_m: float = 45.0
    min_h_m: float = 8.0
    max_h_m: float = 22.0
    min_height_m: float = 10.0
    max_height_m: float = 50.0


@dataclass(frozen=True)
class UrbMaskPropagationConfig:
    # how sharp the LOS<->NLOS transition is (larger => smoother, often more realistic)
    urb_transition_m: float = 22.0

    # optional reflection rescue (kept as a switch)
    enable_refl_gain: bool = True
    gmax_db: float = 10.0
    d0_m: float = 25.0
    refl_beta: float = 0.25


@dataclass(frozen=True)
class UrbMaskScenarioConfig:
    """UrbMask+ config (Gate-B: geometry-driven blockage + optional reflection rescue)."""
    buildings: UrbMaskBuildingsConfig = field(default_factory=UrbMaskBuildingsConfig)
    prop: UrbMaskPropagationConfig = field(default_factory=UrbMaskPropagationConfig)

    def to_manifest(self) -> dict:
        d = asdict(self)
        d["name"] = "UrbMaskPlus"
        return d