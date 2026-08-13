from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


@dataclass(frozen=True)
class TunnelConfig:
    x0_m: float = 1000.0
    x1_m: float = 2600.0
    transition_m: float = 200.0

    # legacy fields kept for CSV backward compatibility
    severity: float = 1.0
    b_floor: float = 0.35
    b_peak: float = 0.45
    bell_gamma: float = 1.6

    # tunnel large-scale propagation
    width_m: float = 16.0
    height_m: float = 7.0
    fc_ghz: float = 5.9
    breakpoint_m: float = 300.0
    near_exp: float = 2.0
    far_exp: float = 1.25
    tunnel_extra_loss_db: float = 0.0
    shadow_sigma_db: float = 2.385

    # measurement-inspired log-distance model for NLoS tunnel links (Wang et al., 2023)
    logdist_eps_db_5p2: float = 50.595
    logdist_n_nlos: float = 1.785
    model_family: str = 'portal_weighted_measured_logdist'

    # channel-induced extra service-time surrogate (kept small and clearly separate
    # from physical channel delay spread; inspired by in-tunnel temporal dispersion studies)
    delay_extra_ms: float = 0.12
    delay_exp_scale_ms: float = 0.25
    shape: str = 'portal_weighted_measured_logdist'

    @staticmethod
    def from_csv(path: Path) -> 'TunnelConfig':
        df = pd.read_csv(path)
        if len(df) < 1:
            raise ValueError(f'Empty tunnel config: {path}')
        r = df.iloc[0]
        return TunnelConfig(
            x0_m=float(r['x0_m']),
            x1_m=float(r['x1_m']),
            transition_m=float(r.get('transition_m', 200.0)),
            severity=float(r.get('severity', 1.0)),
            b_floor=float(r.get('b_floor', 0.35)),
            b_peak=float(r.get('b_peak', 0.45)),
            bell_gamma=float(r.get('bell_gamma', 1.6)),
            width_m=float(r.get('width_m', 16.0)),
            height_m=float(r.get('height_m', 7.0)),
            fc_ghz=float(r.get('fc_ghz', 5.9)),
            breakpoint_m=float(r.get('breakpoint_m', 300.0)),
            near_exp=float(r.get('near_exp', 2.0)),
            far_exp=float(r.get('far_exp', 1.25)),
            tunnel_extra_loss_db=float(r.get('tunnel_extra_loss_db', 0.0)),
            shadow_sigma_db=float(r.get('shadow_sigma_db', 2.385)),
            logdist_eps_db_5p2=float(r.get('logdist_eps_db_5p2', 50.595)),
            logdist_n_nlos=float(r.get('logdist_n_nlos', 1.785)),
            model_family=str(r.get('model_family', r.get('shape', 'portal_weighted_measured_logdist'))),
            delay_extra_ms=float(r.get('delay_extra_ms', 0.12)),
            delay_exp_scale_ms=float(r.get('delay_exp_scale_ms', 0.25)),
            shape=str(r.get('shape', 'portal_weighted_measured_logdist')),
        )

    def to_record(self) -> dict:
        return {
            'x0_m': float(self.x0_m),
            'x1_m': float(self.x1_m),
            'transition_m': float(self.transition_m),
            'severity': float(self.severity),
            'b_floor': float(self.b_floor),
            'b_peak': float(self.b_peak),
            'bell_gamma': float(self.bell_gamma),
            'width_m': float(self.width_m),
            'height_m': float(self.height_m),
            'fc_ghz': float(self.fc_ghz),
            'breakpoint_m': float(self.breakpoint_m),
            'near_exp': float(self.near_exp),
            'far_exp': float(self.far_exp),
            'tunnel_extra_loss_db': float(self.tunnel_extra_loss_db),
            'shadow_sigma_db': float(self.shadow_sigma_db),
            'logdist_eps_db_5p2': float(self.logdist_eps_db_5p2),
            'logdist_n_nlos': float(self.logdist_n_nlos),
            'model_family': str(self.model_family),
            'delay_extra_ms': float(self.delay_extra_ms),
            'delay_exp_scale_ms': float(self.delay_exp_scale_ms),
            'shape': str(self.shape),
        }


def _portal_indicator(x: float, x0: float, x1: float, transition_m: float) -> float:
    t = max(float(transition_m), 1e-6)
    if x < x0:
        return float(np.exp(-((x0 - x) / t) ** 2))
    if x > x1:
        return float(np.exp(-((x - x1) / t) ** 2))
    return 1.0


def _segment_overlap_fraction_1d(xa: float, xb: float, x0: float, x1: float) -> float:
    lo = min(float(xa), float(xb))
    hi = max(float(xa), float(xb))
    seg_len = max(hi - lo, 1e-6)
    ov = max(0.0, min(hi, x1) - max(lo, x0))
    return clamp01(ov / seg_len)


def tunnel_impairment_b(tx_x: float, rx_x: float, cfg: TunnelConfig) -> Tuple[float, float]:
    x0, x1 = float(cfg.x0_m), float(cfg.x1_m)
    L = max(1e-6, x1 - x0)
    mid_x = 0.5 * (float(tx_x) + float(rx_x))
    u = (mid_x - x0) / L

    occ_tx = _portal_indicator(float(tx_x), x0, x1, float(cfg.transition_m))
    occ_rx = _portal_indicator(float(rx_x), x0, x1, float(cfg.transition_m))
    overlap = _segment_overlap_fraction_1d(float(tx_x), float(rx_x), x0, x1)
    soft_occ = 0.5 * (occ_tx + occ_rx)

    b = float(cfg.severity) * clamp01(0.5 * overlap + 0.5 * soft_occ)
    return clamp01(b), float(u)


def tunnel_breakpoint_m(cfg: TunnelConfig) -> float:
    if float(cfg.breakpoint_m) > 0.0:
        return float(cfg.breakpoint_m)
    fc = max(float(cfg.fc_ghz), 0.1)
    lambda_m = 3e8 / (fc * 1e9)
    a_eff = max(min(float(cfg.width_m), float(cfg.height_m)), 1.0)
    return float((a_eff ** 2) / lambda_m)


def tunnel_pathloss_two_slope_db(distance_m: float, cfg: TunnelConfig) -> float:
    d = max(float(distance_m), 1.0)
    fc = max(float(cfg.fc_ghz), 0.1)
    pl1m = 32.4 + 20.0 * np.log10(fc)
    bp = max(1.0, tunnel_breakpoint_m(cfg))
    n1 = max(float(cfg.near_exp), 0.1)
    n2 = max(float(cfg.far_exp), 0.1)

    if d <= bp:
        pl = pl1m + 10.0 * n1 * np.log10(d)
    else:
        pl_bp = pl1m + 10.0 * n1 * np.log10(bp)
        pl = pl_bp + 10.0 * n2 * np.log10(d / bp)

    pl += float(cfg.tunnel_extra_loss_db)
    return float(pl)


def tunnel_pathloss_measured_logdist_db(distance_m: float, cfg: TunnelConfig) -> float:
    # Wang et al. measurements are centered at 5.2 GHz; apply a simple frequency correction
    # when re-targeting to the 5.9 GHz ITS band.
    d = max(float(distance_m), 1.0)
    fc = max(float(cfg.fc_ghz), 0.1)
    eps_db = float(cfg.logdist_eps_db_5p2) + 20.0 * np.log10(fc / 5.2)
    pl = eps_db + 10.0 * float(cfg.logdist_n_nlos) * np.log10(d)
    pl += float(cfg.tunnel_extra_loss_db)
    return float(pl)


def tunnel_pathloss_db(distance_m: float, cfg: TunnelConfig) -> float:
    fam = str(cfg.model_family or cfg.shape).strip().lower()
    if fam in ('two_slope', 'segment_overlap_two_slope', 'portal_weighted_two_slope'):
        return tunnel_pathloss_two_slope_db(distance_m, cfg)
    return tunnel_pathloss_measured_logdist_db(distance_m, cfg)
