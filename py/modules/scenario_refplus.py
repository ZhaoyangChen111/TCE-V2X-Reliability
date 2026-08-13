# modules/scenario_refplus.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field


@dataclass(frozen=True)
class RefPlusGeometryConfig:
    # Day13-style geometry (3 km, two intersections, mild S-curve)
    road_length_m: float = 3000.0
    n_lanes_per_dir: int = 3
    lane_width_m: float = 3.5
    median_gap_m: float = 2.0

    s_curve_x0: float = 900.0
    s_curve_x1: float = 1500.0
    s_curve_amp_y_m: float = 12.0

    i1_x: float = 1000.0
    i2_x: float = 2000.0
    intersection_zone_m: float = 60.0


@dataclass(frozen=True)
class TrafficIDMConfig:
    flow_main_vph: float = 600.0
    veh_length_m: float = 4.5
    min_spawn_gap_m: float = 10.0

    idm_v0_mps: float = 22.0
    idm_T_s: float = 1.2
    idm_a_mps2: float = 1.2
    idm_b_mps2: float = 2.0
    idm_s0_m: float = 2.0
    idm_delta: float = 4.0


@dataclass(frozen=True)
class TrafficSignalConfig:
    sig_cycle_s: float = 90.0
    sig_green_main_s: float = 55.0
    sig_all_red_s: float = 2.0
    sig_offset_i2_s: float = 15.0


@dataclass(frozen=True)
class RefPlusScenarioConfig:
    """RefPlus scenario config (Gate-C: topology + signals + IDM traffic + cross/turn flows)."""
    geom: RefPlusGeometryConfig = field(default_factory=RefPlusGeometryConfig)
    idm: TrafficIDMConfig = field(default_factory=TrafficIDMConfig)
    sig: TrafficSignalConfig = field(default_factory=TrafficSignalConfig)

    # Gate-C: cross road + turning
    cross_enable: bool = True
    flow_cross_vph: float = 400.0
    cross_half_length_m: float = 400.0
    p_turn_i1: float = 0.12
    p_turn_i2: float = 0.12
    p_right: float = 0.80
    p_left: float = 0.20

    def to_manifest(self) -> dict:
        d = asdict(self)
        d["name"] = "RefPlus"
        return d