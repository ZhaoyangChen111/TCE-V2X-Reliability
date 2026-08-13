from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TCEConfig:
    profile: str
    deadline_ms: float
    grace_ms: float
    beta: float
    gamma: float
    msg_rate_hz: float


def _positive_or_default(x: float, default: float) -> float:
    try:
        xv = float(x)
    except Exception:
        return float(default)
    return float(default) if (not np.isfinite(xv) or xv <= 0.0) else float(xv)


def resolve_tce_config(
    profile: str = "safety_awareness",
    deadline_ms: float = 0.0,
    grace_ms: float = 0.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    msg_rate_hz: float = 10.0,
) -> TCEConfig:
    profile0 = (profile or "safety_awareness").strip().lower()

    if profile0 == "pre_crash":
        default_deadline = 20.0
    else:
        default_deadline = 100.0

    message_period_ms = 1000.0 / max(1e-9, float(msg_rate_hz))
    d_eff = _positive_or_default(deadline_ms, default_deadline)
    g_default = min(d_eff, message_period_ms)
    g_eff = _positive_or_default(grace_ms, g_default)
    b_eff = _positive_or_default(beta, math.log(20.0))
    gam_eff = _positive_or_default(gamma, 2.0)

    return TCEConfig(
        profile=profile0,
        deadline_ms=float(d_eff),
        grace_ms=float(g_eff),
        beta=float(b_eff),
        gamma=float(gam_eff),
        msg_rate_hz=float(msg_rate_hz),
    )


def _effective_deadline_series(df: pd.DataFrame, cfg: TCEConfig, prefer_packet_deadline: bool = True) -> np.ndarray:
    """
    When prefer_packet_deadline=True, use the stricter of:
      - the packet-level deadline carried by the raw simulation, and
      - the application-level TCE profile deadline.

    Rationale:
      In this project the raw packet pipeline may carry a looser engineering
      deadline (e.g. 100 ms), while the TCE profile can represent a stricter
      safety semantics (e.g. pre-crash at 20 ms). Using the stricter deadline
      avoids letting a loose packet-level deadline erase the partially-useful
      region that TCE is designed to reveal.
    """
    if prefer_packet_deadline and ("deadline_ms" in df.columns):
        d = pd.to_numeric(df["deadline_ms"], errors="coerce").to_numpy(dtype=float)
        bad = (~np.isfinite(d)) | (d <= 0.0)
        d[bad] = cfg.deadline_ms
        return np.minimum(d, cfg.deadline_ms)
    return np.full(len(df), cfg.deadline_ms, dtype=float)


def compute_tce_utility(
    df: pd.DataFrame,
    cfg: TCEConfig,
    prefer_packet_deadline: bool = True,
) -> pd.DataFrame:
    """
    Important design choice:
      - received_phy_flag comes from raw success_phy if available
      - timely_flag is always recomputed from received_phy_flag + delay_ms + deadline_eff_ms
      - this allows offline re-evaluation under a tighter application deadline
    """
    out = df.copy()

    deadline_eff = _effective_deadline_series(out, cfg=cfg, prefer_packet_deadline=prefer_packet_deadline)
    grace_eff = np.full(len(out), cfg.grace_ms, dtype=float)

    if "success_phy" in out.columns:
        received_phy = pd.to_numeric(out["success_phy"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0.5
    else:
        received_phy = pd.to_numeric(out.get("success", 0), errors="coerce").fillna(0).to_numpy(dtype=float) > 0.5

    delay_ms = pd.to_numeric(out.get("delay_ms", np.nan), errors="coerce").to_numpy(dtype=float)

    timely_flag = received_phy & np.isfinite(delay_ms) & (delay_ms <= deadline_eff)
    tardiness_ms = np.where(received_phy & np.isfinite(delay_ms), np.maximum(delay_ms - deadline_eff, 0.0), np.nan)
    late_received = received_phy & np.isfinite(delay_ms) & (delay_ms > deadline_eff)

    utility = np.zeros(len(out), dtype=float)
    utility[timely_flag] = 1.0

    idx = np.where(late_received & np.isfinite(tardiness_ms))[0]
    if len(idx) > 0:
        tt = tardiness_ms[idx]
        gg = np.maximum(grace_eff[idx], 1e-9)
        x = tt / gg
        u = np.exp(-cfg.beta * np.power(x, cfg.gamma))
        u = np.where(tt <= gg, u, 0.0)
        utility[idx] = np.clip(u, 0.0, 1.0)

    utility_timely = np.where(timely_flag, 1.0, 0.0)
    utility_late = np.where(late_received, utility, 0.0)

    out["deadline_eff_ms"] = deadline_eff
    out["grace_eff_ms"] = grace_eff
    out["tardiness_ms"] = tardiness_ms
    out["received_phy_flag"] = received_phy.astype(int)
    out["timely_flag"] = timely_flag.astype(int)
    out["late_received_flag"] = late_received.astype(int)
    out["utility_timely_full"] = utility_timely.astype(float)
    out["utility_late_partial"] = utility_late.astype(float)
    out["utility_tce"] = utility.astype(float)
    return out


def summarize_tce(df_util: pd.DataFrame) -> Dict[str, Any]:
    n_total = int(len(df_util))

    timely_flag = pd.to_numeric(df_util["timely_flag"], errors="coerce").fillna(0)
    received_phy_flag = pd.to_numeric(df_util["received_phy_flag"], errors="coerce").fillna(0)
    late_received_flag = pd.to_numeric(df_util["late_received_flag"], errors="coerce").fillna(0)

    n_success = int(timely_flag.sum())
    n_success_phy = int(received_phy_flag.sum())
    n_late = int(late_received_flag.sum())

    utility_tce = pd.to_numeric(df_util["utility_tce"], errors="coerce")
    utility_late = pd.to_numeric(df_util["utility_late_partial"], errors="coerce")

    return {
        "n_total": n_total,
        "n_success": n_success,
        "timely_success_rate": float(n_success / n_total) if n_total > 0 else np.nan,
        "n_success_phy": n_success_phy,
        "phy_success_rate": float(n_success_phy / n_total) if n_total > 0 else np.nan,
        "n_late": n_late,
        "late_ratio_phy": float(n_late / max(1, n_success_phy)),
        "tce": float(utility_tce.mean()) if n_total > 0 else np.nan,
        "avg_utility_late_only": float(utility_late[late_received_flag == 1].mean()) if (late_received_flag == 1).any() else 0.0,
        "late_partial_gain": float((utility_tce - timely_flag).clip(lower=0).mean()) if n_total > 0 else np.nan,
    }


def aggregate_tce_by_group(df_util: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    has_cbr = "cbr" in df_util.columns
    has_link_bias = "link_bias" in df_util.columns
    has_hotspot = "hotspot_mult_col" in df_util.columns

    for key, g in df_util.groupby(group_col, dropna=True, sort=True):
        s = summarize_tce(g)
        rows.append(
            {
                group_col: key,
                **s,
                "nlos_ratio": float((g["link_state"] == "NLOS").mean()) if "link_state" in g.columns else np.nan,
                "tunnel_ratio": float((g["link_state"] == "TUNNEL").mean()) if "link_state" in g.columns else np.nan,
                "avg_blockage_b": float(pd.to_numeric(g["blockage_b"], errors="coerce").mean()) if "blockage_b" in g.columns else np.nan,
                "avg_n_cs": float(pd.to_numeric(g["n_cs"], errors="coerce").mean()) if "n_cs" in g.columns else np.nan,
                "avg_cbr": float(pd.to_numeric(g["cbr"], errors="coerce").mean()) if has_cbr else np.nan,
                "avg_p_col": float(pd.to_numeric(g["p_col"], errors="coerce").mean()) if "p_col" in g.columns else np.nan,
                "avg_cong_delay_ms": float(pd.to_numeric(g["cong_delay_ms"], errors="coerce").mean()) if "cong_delay_ms" in g.columns else np.nan,
                "avg_link_bias": float(pd.to_numeric(g["link_bias"], errors="coerce").mean()) if has_link_bias else np.nan,
                "avg_hotspot_mult_col": float(pd.to_numeric(g["hotspot_mult_col"], errors="coerce").mean()) if has_hotspot else np.nan,
            }
        )
    return pd.DataFrame(rows)