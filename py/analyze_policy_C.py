from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paths_C import ensure_run_dirs_a, make_run_id, load_latest_run_id, default_run_prefix
from run_logging import log_command, update_manifest
from modules.tce_metric import resolve_tce_config, compute_tce_utility, summarize_tce


def _pick_run_id(arg_run_id: str) -> str:
    s = (arg_run_id or "").strip()
    if s == "":
        return make_run_id(prefix=default_run_prefix())
    if s.lower() == "latest":
        rid = load_latest_run_id()
        return rid if rid else make_run_id(prefix=default_run_prefix())
    return s


def _pick_latest_packets_file(raw_dir: Path, scenario: str, ret: int, policy_tag: str) -> tuple[Path, str, str]:
    cands = list(raw_dir.glob(f"results_packets__{scenario}__ret{ret}__{policy_tag}__seed*.csv"))
    if not cands:
        raise FileNotFoundError(
            f"Cannot find policy packets in {raw_dir} for scenario={scenario} ret={ret} policy_tag={policy_tag}"
        )
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    p = cands[0]
    full_tag = p.stem.split(f"__ret{ret}__")[-1]
    seed_tag = p.stem.split(f"__{policy_tag}__")[-1]
    return p, full_tag, seed_tag


def _pick_latest_decision_file(raw_dir: Path, scenario: str, ret: int, policy_tag: str) -> Path | None:
    cands = list(raw_dir.glob(f"results_retx_decisions__{scenario}__ret{ret}__{policy_tag}__seed*.csv"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _ci95(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) <= 1:
        return np.nan
    return float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))


def _safe_num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _mean_or_nan(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(x.mean()) if len(x) > 0 else np.nan


def _share_abs_le(x: pd.Series, bound: float) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float((x.abs() <= float(bound)).mean())


def _share_str_eq(x: pd.Series, value: str) -> float:
    if x is None or len(x) == 0:
        return np.nan
    s = x.astype(str).str.strip().str.lower()
    target = str(value).strip().lower()
    valid = s.notna()
    if not bool(valid.any()):
        return np.nan
    return float((s == target).mean())


def _dist_mask(df: pd.DataFrame, lo: float | None = None, hi: float | None = None) -> pd.Series:
    if "distance_m" not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)
    dist = pd.to_numeric(df["distance_m"], errors="coerce")
    mask = dist.notna()
    if lo is not None:
        mask &= dist >= float(lo)
    if hi is not None:
        mask &= dist < float(hi)
    return mask


def _seed_packet_metrics(g: pd.DataFrame, g_tce: pd.DataFrame | None = None) -> dict:
    success_raw = _safe_num(g, "success", default=0.0).fillna(0.0)
    success_phy_raw = _safe_num(g, "success_phy", default=0.0).fillna(0.0)
    late_raw = _safe_num(g, "late", default=0.0).fillna(0.0)
    delay_ms_raw = _safe_num(g, "delay_ms", default=np.nan)
    attempts = _safe_num(g, "n_tx_attempts", default=np.nan)

    out = {
        "n_packets": int(len(g)),
        # raw packet-level metrics kept for audit/debug only
        "raw_success_rate": _mean_or_nan(success_raw),
        "raw_phy_rate": _mean_or_nan(success_phy_raw),
        "raw_late_ratio_total": _mean_or_nan(late_raw),
        "raw_late_ratio_phy": float(late_raw.sum() / max(1.0, float(success_phy_raw.sum()))) if len(g) > 0 else np.nan,
        "raw_avg_delay_success_ms": float(delay_ms_raw[success_raw > 0.5].mean()) if (success_raw > 0.5).any() else np.nan,
        "raw_avg_delay_phy_ms": float(delay_ms_raw[success_phy_raw > 0.5].mean()) if (success_phy_raw > 0.5).any() else np.nan,
        # study-aligned metrics (will be overwritten from TCE view when available)
        "timely_rate": _mean_or_nan(success_raw),
        "phy_rate": _mean_or_nan(success_phy_raw),
        "late_ratio_total": _mean_or_nan(late_raw),
        "late_ratio_phy": float(late_raw.sum() / max(1.0, float(success_phy_raw.sum()))) if len(g) > 0 else np.nan,
        "avg_attempts": _mean_or_nan(attempts),
        "avg_delay_success_ms": float(delay_ms_raw[success_raw > 0.5].mean()) if (success_raw > 0.5).any() else np.nan,
        "avg_delay_phy_ms": float(delay_ms_raw[success_phy_raw > 0.5].mean()) if (success_phy_raw > 0.5).any() else np.nan,
        "tce": np.nan,
        "late_partial_gain": np.nan,
        "avg_utility_late_only": np.nan,
        "tce_timely_rate": np.nan,
        "tce_phy_rate": np.nan,
        "n_total": np.nan,
        "n_late": np.nan,
        "hidden_gap_phy_minus_timely": np.nan,
        "hidden_gap_tce_minus_timely": np.nan,
    }

    if g_tce is not None and len(g_tce) > 0:
        s = summarize_tce(g_tce)

        timely_flag = _safe_num(g_tce, "timely_flag", default=0.0).fillna(0.0)
        received_phy_flag = _safe_num(g_tce, "received_phy_flag", default=0.0).fillna(0.0)
        delay_ms_tce = _safe_num(g_tce, "delay_ms", default=np.nan)

        timely_rate = float(s["timely_success_rate"])
        phy_rate = float(s["phy_success_rate"])
        tce = float(s["tce"])
        n_total = float(s["n_total"])
        n_late = float(s["n_late"])

        out.update(
            {
                # primary policy-study metrics must match TCE / readiness line
                "timely_rate": timely_rate,
                "phy_rate": phy_rate,
                "late_ratio_total": float(n_late / max(1.0, n_total)),
                "late_ratio_phy": float(s["late_ratio_phy"]),
                "avg_delay_success_ms": float(delay_ms_tce[timely_flag > 0.5].mean()) if (timely_flag > 0.5).any() else np.nan,
                "avg_delay_phy_ms": float(delay_ms_tce[received_phy_flag > 0.5].mean()) if (received_phy_flag > 0.5).any() else np.nan,
                "tce": tce,
                "late_partial_gain": float(s["late_partial_gain"]),
                "avg_utility_late_only": float(s["avg_utility_late_only"]),
                "tce_timely_rate": timely_rate,
                "tce_phy_rate": phy_rate,
                "n_total": n_total,
                "n_late": n_late,
                "hidden_gap_phy_minus_timely": phy_rate - timely_rate,
                "hidden_gap_tce_minus_timely": tce - timely_rate,
            }
        )

    return out


def _seed_decision_metrics(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return {
            "n_decisions": 0,
            "retx_rate": np.nan,
            "avg_expected_gain": np.nan,
            "avg_cost_ci": np.nan,
            "avg_gain_over_cost": np.nan,
            "avg_score": np.nan,
            "avg_incremental_delay_ms": np.nan,
            "avg_chain_expected_gain": np.nan,
            "avg_chain_expected_cost": np.nan,
            "avg_chain_gain_over_cost": np.nan,
            "avg_chain_score": np.nan,
            "avg_delay_norm": np.nan,
            "avg_airtime_norm": np.nan,
            "avg_delay_term": np.nan,
            "avg_airtime_term": np.nan,
            "avg_resource_term": np.nan,
            "avg_cost_multiplier": np.nan,
            "avg_predicted_next_cbr": np.nan,
            "avg_predicted_next_p_col": np.nan,
            "avg_predicted_busy_pressure": np.nan,
            "avg_mdp_model_hit": np.nan,
            "avg_mdp_model_miss": np.nan,
            "avg_mdp_exact_hit": np.nan,
            "avg_mdp_coarse_hit": np.nan,
            "avg_mdp_global_default_hit": np.nan,
            "avg_mdp_chain_fallback_used": np.nan,
            "avg_mdp_fallback_used": np.nan,
            "share_mdp_decision_source_model": np.nan,
            "share_mdp_decision_source_chain_fallback": np.nan,
            "share_mdp_decision_source_drop": np.nan,
            "avg_mdp_model_samples": np.nan,
            "avg_mdp_effective_min_samples": np.nan,
            "avg_mdp_lookup_rank": np.nan,
            "avg_mdp_q_continue": np.nan,
            "avg_mdp_q_stop": np.nan,
            "avg_mdp_cost_scale": np.nan,
            "avg_mdp_cost_raw": np.nan,
            "avg_mdp_cost_scaled": np.nan,
            "avg_mdp_expected_success_term": np.nan,
            "avg_mdp_future_fail_term": np.nan,
            "avg_mdp_raw_margin": np.nan,
            "avg_mdp_threshold_applied": np.nan,
            "avg_mdp_thresholded_margin": np.nan,
            "avg_mdp_chain_score_used": np.nan,
            "share_mdp_abs_raw_margin_le_0p02": np.nan,
            "share_mdp_abs_thresholded_margin_le_0p02": np.nan,
            "avg_mdp_value": np.nan,
            "avg_mdp_depth_used": np.nan,
        }
    return {
        "n_decisions": int(len(g)),
        "retx_rate": _mean_or_nan(_safe_num(g, "decision_retransmit", default=np.nan)),
        "avg_expected_gain": _mean_or_nan(_safe_num(g, "expected_gain", default=np.nan)),
        "avg_cost_ci": _mean_or_nan(_safe_num(g, "cost_ci", default=np.nan)),
        "avg_gain_over_cost": _mean_or_nan(_safe_num(g, "gain_over_cost", default=np.nan)),
        "avg_score": _mean_or_nan(_safe_num(g, "score", default=np.nan)),
        "avg_incremental_delay_ms": _mean_or_nan(_safe_num(g, "incremental_delay_ms", default=np.nan)),
        "avg_chain_expected_gain": _mean_or_nan(_safe_num(g, "chain_expected_gain", default=np.nan)),
        "avg_chain_expected_cost": _mean_or_nan(_safe_num(g, "chain_expected_cost", default=np.nan)),
        "avg_chain_gain_over_cost": _mean_or_nan(_safe_num(g, "chain_gain_over_cost", default=np.nan)),
        "avg_chain_score": _mean_or_nan(_safe_num(g, "chain_score", default=np.nan)),
        "avg_delay_norm": _mean_or_nan(_safe_num(g, "delay_norm", default=np.nan)),
        "avg_airtime_norm": _mean_or_nan(_safe_num(g, "airtime_norm", default=np.nan)),
        "avg_delay_term": _mean_or_nan(_safe_num(g, "delay_term", default=np.nan)),
        "avg_airtime_term": _mean_or_nan(_safe_num(g, "airtime_term", default=np.nan)),
        "avg_resource_term": _mean_or_nan(_safe_num(g, "resource_term", default=np.nan)),
        "avg_cost_multiplier": _mean_or_nan(_safe_num(g, "cost_multiplier", default=np.nan)),
        "avg_predicted_next_cbr": _mean_or_nan(_safe_num(g, "predicted_next_cbr", default=np.nan)),
        "avg_predicted_next_p_col": _mean_or_nan(_safe_num(g, "predicted_next_p_col", default=np.nan)),
        "avg_predicted_busy_pressure": _mean_or_nan(_safe_num(g, "predicted_busy_pressure", default=np.nan)),
        "avg_mdp_model_hit": _mean_or_nan(_safe_num(g, "mdp_model_hit", default=np.nan)),
        "avg_mdp_model_miss": _mean_or_nan(_safe_num(g, "mdp_model_miss", default=np.nan)),
        "avg_mdp_exact_hit": _mean_or_nan(_safe_num(g, "mdp_exact_hit", default=np.nan)),
        "avg_mdp_coarse_hit": _mean_or_nan(_safe_num(g, "mdp_coarse_hit", default=np.nan)),
        "avg_mdp_global_default_hit": _mean_or_nan(_safe_num(g, "mdp_global_default_hit", default=np.nan)),
        "avg_mdp_chain_fallback_used": _mean_or_nan(_safe_num(g, "mdp_chain_fallback_used", default=np.nan)),
        "avg_mdp_fallback_used": _mean_or_nan(_safe_num(g, "mdp_fallback_used", default=np.nan)),
        "share_mdp_decision_source_model": _share_str_eq(g.get("mdp_decision_source", pd.Series(index=g.index, dtype=object)), "model"),
        "share_mdp_decision_source_chain_fallback": _share_str_eq(g.get("mdp_decision_source", pd.Series(index=g.index, dtype=object)), "chain_fallback"),
        "share_mdp_decision_source_drop": _share_str_eq(g.get("mdp_decision_source", pd.Series(index=g.index, dtype=object)), "drop"),
        "avg_mdp_model_samples": _mean_or_nan(_safe_num(g, "mdp_model_samples", default=np.nan)),
        "avg_mdp_effective_min_samples": _mean_or_nan(_safe_num(g, "mdp_effective_min_samples", default=np.nan)),
        "avg_mdp_lookup_rank": _mean_or_nan(_safe_num(g, "mdp_lookup_rank", default=np.nan)),
        "avg_mdp_q_continue": _mean_or_nan(_safe_num(g, "mdp_q_continue", default=np.nan)),
        "avg_mdp_q_stop": _mean_or_nan(_safe_num(g, "mdp_q_stop", default=np.nan)),
        "avg_mdp_cost_scale": _mean_or_nan(_safe_num(g, "mdp_cost_scale", default=np.nan)),
        "avg_mdp_cost_raw": _mean_or_nan(_safe_num(g, "mdp_cost_raw", default=np.nan)),
        "avg_mdp_cost_scaled": _mean_or_nan(_safe_num(g, "mdp_cost_scaled", default=np.nan)),
        "avg_mdp_expected_success_term": _mean_or_nan(_safe_num(g, "mdp_expected_success_term", default=np.nan)),
        "avg_mdp_future_fail_term": _mean_or_nan(_safe_num(g, "mdp_future_fail_term", default=np.nan)),
        "avg_mdp_raw_margin": _mean_or_nan(_safe_num(g, "mdp_raw_margin", default=np.nan)),
        "avg_mdp_threshold_applied": _mean_or_nan(_safe_num(g, "mdp_threshold_applied", default=np.nan)),
        "avg_mdp_thresholded_margin": _mean_or_nan(_safe_num(g, "mdp_thresholded_margin", default=np.nan)),
        "avg_mdp_chain_score_used": _mean_or_nan(_safe_num(g, "mdp_chain_score_used", default=np.nan)),
        "share_mdp_abs_raw_margin_le_0p02": _share_abs_le(_safe_num(g, "mdp_raw_margin", default=np.nan), 0.02),
        "share_mdp_abs_thresholded_margin_le_0p02": _share_abs_le(_safe_num(g, "mdp_thresholded_margin", default=np.nan), 0.02),
        "avg_mdp_value": _mean_or_nan(_safe_num(g, "mdp_value", default=np.nan)),
        "avg_mdp_depth_used": _mean_or_nan(_safe_num(g, "mdp_depth_used", default=np.nan)),
    }


def _parse_band_edges(s: str) -> list[float]:
    vals = []
    for part in str(s or "").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    vals = sorted(set(vals))
    return vals if len(vals) >= 2 else [0.0, 50.0, 100.0, 150.0, 200.0]


def _band_label(lo: float, hi: float) -> str:
    return f"{int(round(lo))}-{int(round(hi))}m"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=str, default="latest")
    ap.add_argument("--scenario", required=True, choices=["Ref", "UrbMask", "Tunnel"])
    ap.add_argument("--retrans", required=True, type=int, choices=[0, 1, 2])
    ap.add_argument("--policy_tag", required=True, type=str)
    ap.add_argument("--tce_profile", type=str, default="custom", choices=["safety_awareness", "pre_crash", "custom"])
    ap.add_argument("--tce_deadline_ms", type=float, default=0.0)
    ap.add_argument("--tce_grace_ms", type=float, default=0.0)
    ap.add_argument("--tce_beta", type=float, default=0.0)
    ap.add_argument("--tce_gamma", type=float, default=0.0)
    ap.add_argument("--msg_rate_hz", type=float, default=10.0)
    ap.add_argument(
        "--max_distance_m",
        type=float,
        default=200.0,
        help="Only keep packets/decisions with distance_m <= this value. Use <=0 to disable.",
    )
    ap.add_argument("--prefer_packet_deadline", action="store_true")
    ap.add_argument("--band_edges_m", type=str, default="0,50,100,150,200")
    args = ap.parse_args()

    run_id = _pick_run_id(args.run_id)
    rp = ensure_run_dirs_a(run_id, save_as_latest=False, meta={"script": "analyze_policy_C.py"})
    log_command(
        run_id,
        rp.run_results_dir,
        extra=f"analyze_policy scenario={args.scenario} ret={args.retrans} policy={args.policy_tag}",
    )

    pkt_path, _full_tag, seed_tag = _pick_latest_packets_file(rp.raw_dir, args.scenario, args.retrans, args.policy_tag)
    dec_path = _pick_latest_decision_file(rp.raw_dir, args.scenario, args.retrans, args.policy_tag)

    pkt = pd.read_csv(pkt_path)
    n_raw_pkt = int(len(pkt))
    if float(args.max_distance_m) > 0.0 and len(pkt) > 0:
        keep = _dist_mask(pkt, hi=float(args.max_distance_m) + 1e-12)
        pkt = pkt.loc[keep].copy()
        print(f"[INFO] policy packets distance filter <= {float(args.max_distance_m):.1f} m: kept {len(pkt)}/{n_raw_pkt} packets")

    tce_cfg = resolve_tce_config(
        profile=str(args.tce_profile),
        deadline_ms=float(args.tce_deadline_ms),
        grace_ms=float(args.tce_grace_ms),
        beta=float(args.tce_beta),
        gamma=float(args.tce_gamma),
        msg_rate_hz=float(args.msg_rate_hz),
    )
    pkt_tce = compute_tce_utility(pkt, cfg=tce_cfg, prefer_packet_deadline=bool(args.prefer_packet_deadline))

    if dec_path is not None:
        try:
            dec = pd.read_csv(dec_path)
            n_raw_dec = int(len(dec))
            if float(args.max_distance_m) > 0.0 and len(dec) > 0:
                keepd = _dist_mask(dec, hi=float(args.max_distance_m) + 1e-12)
                dec = dec.loc[keepd].copy()
                print(f"[INFO] policy decisions distance filter <= {float(args.max_distance_m):.1f} m: kept {len(dec)}/{n_raw_dec} rows")
        except pd.errors.EmptyDataError:
            dec = pd.DataFrame()
    else:
        dec = pd.DataFrame()

    if len(pkt) == 0:
        raise RuntimeError(
            f"Packets file is empty after filtering: {pkt_path}. "
            "Policy analysis refuses to produce an empty success summary."
        )

    packet_rows = []
    for seed, g in pkt.groupby("seed", sort=True):
        gt = pkt_tce[pkt_tce["seed"] == seed].copy() if "seed" in pkt_tce.columns else None
        row = {
            "scenario": str(args.scenario),
            "retrans": int(args.retrans),
            "policy_tag": str(args.policy_tag),
            "retx_policy": str(g["retx_policy"].iloc[0]) if "retx_policy" in g.columns else "classical",
            "seed": int(seed),
            **_seed_packet_metrics(g, gt),
        }
        gd = dec[dec["seed"] == seed].copy() if (len(dec) > 0 and "seed" in dec.columns) else pd.DataFrame()
        row.update(_seed_decision_metrics(gd))
        packet_rows.append(row)

    by_seed = pd.DataFrame(packet_rows).sort_values("seed").reset_index(drop=True)
    by_seed_path = rp.tables_dir / f"policy_by_seed__{args.scenario}__ret{args.retrans}__{args.policy_tag}__{seed_tag}.csv"
    by_seed.to_csv(by_seed_path, index=False)

    summary_cols = [
        "timely_rate",
        "phy_rate",
        "late_ratio_total",
        "late_ratio_phy",
        "avg_attempts",
        "avg_delay_success_ms",
        "avg_delay_phy_ms",
        "tce",
        "late_partial_gain",
        "avg_utility_late_only",
        "tce_timely_rate",
        "tce_phy_rate",
        "n_total",
        "n_late",
        "hidden_gap_phy_minus_timely",
        "hidden_gap_tce_minus_timely",
        "raw_success_rate",
        "raw_phy_rate",
        "raw_late_ratio_total",
        "raw_late_ratio_phy",
        "raw_avg_delay_success_ms",
        "raw_avg_delay_phy_ms",
        "retx_rate",
        "avg_expected_gain",
        "avg_cost_ci",
        "avg_gain_over_cost",
        "avg_score",
        "avg_incremental_delay_ms",
        "avg_chain_expected_gain",
        "avg_chain_expected_cost",
        "avg_chain_gain_over_cost",
        "avg_chain_score",
        "avg_delay_norm",
        "avg_airtime_norm",
        "avg_delay_term",
        "avg_airtime_term",
        "avg_resource_term",
        "avg_cost_multiplier",
        "avg_predicted_next_cbr",
        "avg_predicted_next_p_col",
        "avg_predicted_busy_pressure",
        "avg_mdp_model_hit",
        "avg_mdp_model_miss",
        "avg_mdp_exact_hit",
        "avg_mdp_coarse_hit",
        "avg_mdp_global_default_hit",
        "avg_mdp_chain_fallback_used",
        "avg_mdp_fallback_used",
        "share_mdp_decision_source_model",
        "share_mdp_decision_source_chain_fallback",
        "share_mdp_decision_source_drop",
        "avg_mdp_model_samples",
        "avg_mdp_effective_min_samples",
        "avg_mdp_lookup_rank",
        "avg_mdp_q_continue",
        "avg_mdp_q_stop",
        "avg_mdp_cost_scale",
        "avg_mdp_cost_raw",
        "avg_mdp_cost_scaled",
        "avg_mdp_expected_success_term",
        "avg_mdp_future_fail_term",
        "avg_mdp_raw_margin",
        "avg_mdp_threshold_applied",
        "avg_mdp_thresholded_margin",
        "avg_mdp_chain_score_used",
        "share_mdp_abs_raw_margin_le_0p02",
        "share_mdp_abs_thresholded_margin_le_0p02",
        "avg_mdp_value",
        "avg_mdp_depth_used",
    ]
    summary = {
        "scenario": str(args.scenario),
        "retrans": int(args.retrans),
        "policy_tag": str(args.policy_tag),
        "retx_policy": str(by_seed["retx_policy"].iloc[0]) if len(by_seed) > 0 else "unknown",
        "n_seeds": int(len(by_seed)),
        "analysis_max_distance_m": float(args.max_distance_m),
        "packets_file": pkt_path.name,
        "decisions_file": dec_path.name if dec_path is not None else None,
        "tce_profile": str(args.tce_profile),
        "tce_deadline_ms": float(tce_cfg.deadline_ms),
        "tce_grace_ms": float(tce_cfg.grace_ms),
        "tce_beta": float(tce_cfg.beta),
        "tce_gamma": float(tce_cfg.gamma),
    }
    for col in summary_cols:
        if col in by_seed.columns:
            summary[f"mean_{col}"] = _mean_or_nan(by_seed[col]) if len(by_seed) > 0 else np.nan
            summary[f"ci95_{col}"] = _ci95(by_seed[col]) if len(by_seed) > 1 else np.nan

    summary_df = pd.DataFrame([summary])
    summary_path = rp.tables_dir / f"policy_summary__{args.scenario}__ret{args.retrans}__{args.policy_tag}__{seed_tag}.csv"
    summary_df.to_csv(summary_path, index=False)

    band_edges = _parse_band_edges(args.band_edges_m)
    band_rows = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        gp = pkt.loc[_dist_mask(pkt, lo=float(lo), hi=float(hi))].copy()
        if len(gp) == 0:
            continue
        gt = pkt_tce.loc[_dist_mask(pkt_tce, lo=float(lo), hi=float(hi))].copy()
        gd = dec.loc[_dist_mask(dec, lo=float(lo), hi=float(hi))].copy() if len(dec) > 0 else pd.DataFrame()
        row = {
            "scenario": str(args.scenario),
            "retrans": int(args.retrans),
            "policy_tag": str(args.policy_tag),
            "retx_policy": str(gp["retx_policy"].iloc[0]) if "retx_policy" in gp.columns else "classical",
            "band_label": _band_label(float(lo), float(hi)),
            "dist_lo_m": float(lo),
            "dist_hi_m": float(hi),
            **_seed_packet_metrics(gp, gt),
            **_seed_decision_metrics(gd),
        }
        band_rows.append(row)

    band_path = rp.tables_dir / f"policy_by_band__{args.scenario}__ret{args.retrans}__{args.policy_tag}__{seed_tag}.csv"
    pd.DataFrame(band_rows).to_csv(band_path, index=False)

    update_manifest(
        rp.manifest_path,
        {
            "last_policy_analyze": {
                "scenario": str(args.scenario),
                "retrans": int(args.retrans),
                "policy_tag": str(args.policy_tag),
                "packets_file": pkt_path.name,
                "decisions_file": dec_path.name if dec_path is not None else None,
                "outputs": [by_seed_path.name, summary_path.name, band_path.name],
                "tce_profile": str(args.tce_profile),
                "tce_deadline_ms": float(tce_cfg.deadline_ms),
                "tce_grace_ms": float(tce_cfg.grace_ms),
                "tce_beta": float(tce_cfg.beta),
                "tce_gamma": float(tce_cfg.gamma),
                "analysis_max_distance_m": float(args.max_distance_m),
            }
        },
    )

    print(f"[OK] run_id={run_id}")
    print(f"[OK] packets -> {pkt_path.name}")
    if dec_path is not None:
        print(f"[OK] decisions -> {dec_path.name}")
    print(f"[OK] policy_by_seed -> {by_seed_path.name}")
    print(f"[OK] policy_summary -> {summary_path.name}")
    print(f"[OK] policy_by_band -> {band_path.name}")


if __name__ == "__main__":
    main()
