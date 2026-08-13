from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from modules.traffic_idm import IDMParams


@dataclass(frozen=True)
class VehicleProfile:
    veh_type: str
    veh_length_m: float
    idm_params: IDMParams


def _jitter(rng: np.random.Generator, x: float, rel_sigma: float, lo: float, hi: float) -> float:
    if rel_sigma <= 1e-12:
        return float(np.clip(float(x), lo, hi))
    y = float(x) * (1.0 + float(rng.normal(0.0, rel_sigma)))
    return float(np.clip(y, lo, hi))


def sample_vehicle_profile(
    rng: np.random.Generator,
    base_params: IDMParams,
    base_length_m: float = 4.5,
    *,
    enable_vehicle_mix: bool = True,
    car_ratio: float = 0.85,
    heavy_length_m: float = 11.5,
    car_speed_scale: float = 1.00,
    heavy_speed_scale: float = 0.82,
    car_headway_scale: float = 1.00,
    heavy_headway_scale: float = 1.18,
    car_accel_scale: float = 1.00,
    heavy_accel_scale: float = 0.72,
    car_brake_scale: float = 1.00,
    heavy_brake_scale: float = 0.86,
) -> VehicleProfile:
    """
    Lightweight heterogeneous vehicle sampler.

    Goal:
      - keep your current IDM structure
      - introduce only mild heterogeneity (car vs heavy)
      - preserve numerical stability
    """
    if not enable_vehicle_mix:
        veh_type = "car"
    else:
        veh_type = "car" if float(rng.random()) < float(car_ratio) else "heavy"

    if veh_type == "car":
        veh_length = _jitter(rng, float(base_length_m), 0.06, 3.8, 6.0)
        idm = IDMParams(
            v0_mps=_jitter(rng, float(base_params.v0_mps) * float(car_speed_scale), 0.08, 8.0, 42.0),
            T_s=_jitter(rng, float(base_params.T_s) * float(car_headway_scale), 0.10, 0.8, 2.2),
            a_mps2=_jitter(rng, float(base_params.a_mps2) * float(car_accel_scale), 0.10, 0.6, 3.0),
            b_mps2=_jitter(rng, float(base_params.b_mps2) * float(car_brake_scale), 0.08, 1.0, 4.5),
            s0_m=_jitter(rng, float(base_params.s0_m), 0.08, 1.0, 4.0),
            delta=float(base_params.delta),
        )
    else:
        veh_length = _jitter(rng, float(heavy_length_m), 0.08, 8.0, 16.0)
        idm = IDMParams(
            v0_mps=_jitter(rng, float(base_params.v0_mps) * float(heavy_speed_scale), 0.08, 6.0, 32.0),
            T_s=_jitter(rng, float(base_params.T_s) * float(heavy_headway_scale), 0.10, 1.0, 3.0),
            a_mps2=_jitter(rng, float(base_params.a_mps2) * float(heavy_accel_scale), 0.10, 0.4, 2.0),
            b_mps2=_jitter(rng, float(base_params.b_mps2) * float(heavy_brake_scale), 0.08, 0.8, 3.5),
            s0_m=_jitter(rng, float(base_params.s0_m) * 1.10, 0.08, 1.5, 5.0),
            delta=float(base_params.delta),
        )

    return VehicleProfile(
        veh_type=str(veh_type),
        veh_length_m=float(veh_length),
        idm_params=idm,
    )