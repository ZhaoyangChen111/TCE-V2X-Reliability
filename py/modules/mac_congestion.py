from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CongestionParams:
    """
    Lightweight but stable congestion proxy.
    """

    r_cs_m: float = 150.0
    min_speed_mps: float = 0.0

    alpha_col: float = 0.012
    beta_delay_ms: float = 0.10
    exp_scale_ms: float = 0.10

    pkt_bytes: int = 300
    phy_rate_mbps: float = 6.0
    mac_efficiency: float = 0.55
    phy_overhead_us: float = 300.0

    gamma_cbr_col: float = 0.40
    gamma_cbr_delay: float = 0.60

    cbr_cap: float = 0.75
    p_col_cap: float = 0.85
    max_extra_delay_ms: float = 150.0


def compute_airtime_s(
    pkt_bytes: int,
    phy_rate_mbps: float,
    mac_efficiency: float = 0.55,
    phy_overhead_us: float = 300.0,
) -> float:
    b = max(1, int(pkt_bytes))
    r = max(0.1, float(phy_rate_mbps)) * 1e6
    eff = float(np.clip(float(mac_efficiency), 0.05, 0.95))
    payload_s = (b * 8.0) / (r * eff)
    overhead_s = max(0.0, float(phy_overhead_us)) * 1e-6
    return float(payload_s + overhead_s)


def compute_cbr(
    n_cs: int,
    msg_rate_hz: float,
    airtime_s: float,
    cbr_cap: float = 0.75,
) -> float:
    n = max(1, int(n_cs))
    n_others = max(0, n - 1)
    rate = max(0.0, float(msg_rate_hz))
    at = max(0.0, float(airtime_s))
    busy = float(n_others) * rate * at
    return float(np.clip(busy, 0.0, float(cbr_cap)))


def _effective_contenders(n_cs: int) -> float:
    n = max(1, int(n_cs))
    return float(np.log1p(max(0, n - 1)))


def p_collision_from_ncs(
    n_cs: int,
    alpha_col: float,
    cbr: Optional[float] = None,
    gamma_cbr_col: float = 0.40,
    p_col_cap: float = 0.85,
) -> float:
    a = max(0.0, float(alpha_col))
    n_eff = _effective_contenders(n_cs)
    p_base = 1.0 - float(np.exp(-a * n_eff))

    if cbr is None:
        return float(np.clip(p_base, 0.0, float(p_col_cap)))

    c = float(np.clip(float(cbr), 0.0, 0.99))
    busy_term = 1.0 - float(np.exp(-max(0.0, float(gamma_cbr_col)) * c))
    p = p_base + (1.0 - p_base) * busy_term
    return float(np.clip(p, 0.0, float(p_col_cap)))


def congestion_extra_delay_ms(
    rng: np.random.Generator,
    n_cs: int,
    beta_delay_ms: float,
    exp_scale_ms: float,
    cbr: Optional[float] = None,
    gamma_cbr_delay: float = 0.60,
    max_extra_delay_ms: float = 150.0,
) -> float:
    if int(n_cs) <= 1:
        return 0.0

    beta = max(0.0, float(beta_delay_ms))
    scale = max(0.0, float(exp_scale_ms))
    n_eff = _effective_contenders(n_cs)

    amp = 1.0
    if cbr is not None:
        c = float(np.clip(float(cbr), 0.0, 0.99))
        amp = 1.0 + max(0.0, float(gamma_cbr_delay)) * c

    det = beta * n_eff * amp
    tail = float(rng.exponential(scale=scale * max(1.0, n_eff) * amp)) if scale > 1e-12 else 0.0
    out = det + tail
    return float(np.clip(out, 0.0, float(max_extra_delay_ms)))


def compute_ncs_from_distances(
    dist_all: np.ndarray,
    tx_index: int,
    r_cs_m: float,
    active_mask: Optional[np.ndarray] = None,
    speed_all: Optional[np.ndarray] = None,
    min_speed_mps: float = 0.0,
) -> int:
    V = int(dist_all.shape[0])
    if active_mask is None:
        active_mask = np.isfinite(dist_all)

    within = (dist_all <= float(r_cs_m)) & active_mask

    others = within.copy()
    if 0 <= int(tx_index) < V:
        others[int(tx_index)] = False

    if speed_all is not None and float(min_speed_mps) > 0.0:
        others = others & (speed_all >= float(min_speed_mps))

    n_others = int(np.sum(others))
    return 1 + n_others