# modules/traffic_idm.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class IDMParams:
    v0_mps: float = 22.0     # desired speed
    T_s: float = 1.2         # time headway
    a_mps2: float = 1.2      # max accel
    b_mps2: float = 2.0      # comfortable decel
    s0_m: float = 2.0        # min gap
    delta: float = 4.0       # accel exponent


def idm_accel(v: np.ndarray, dv: np.ndarray, gap: np.ndarray, p: IDMParams) -> np.ndarray:
    """
    IDM acceleration. All arrays are element-wise.
    dv = v - v_lead
    gap must be > 0.
    """
    v0 = float(p.v0_mps)
    T = float(p.T_s)
    a = float(p.a_mps2)
    b = float(p.b_mps2)
    s0 = float(p.s0_m)
    delta = float(p.delta)

    gap = np.maximum(gap, 0.1)
    v = np.maximum(v, 0.0)

    s_star = s0 + v * T + (v * dv) / (2.0 * np.sqrt(max(a * b, 1e-6)))
    accel = a * (1.0 - (v / max(v0, 1e-6)) ** delta - (s_star / gap) ** 2)
    # avoid insane spikes
    accel = np.clip(accel, -4.0 * b, 2.0 * a)
    return accel