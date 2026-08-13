from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paths_C import ensure_run_dirs_a, make_run_id, load_latest_run_id, default_run_prefix
from run_logging import log_command, update_manifest
from modules.tce_metric import resolve_tce_config, compute_tce_utility, summarize_tce, aggregate_tce_by_group


def _pick_run_id(arg_run_id: str) -> str:
    s = (arg_run_id or "").strip()
    if s == "":
        return make_run_id(prefix=default_run_prefix())
    if s.lower() == "latest":
        rid = load_latest_run_id()
        return rid if rid else make_run_id(prefix=default_run_prefix())
    return s


def _pick_latest_packets_file(raw_dir: Path, scenario: str, ret: int, policy_tag: str = "") -> tuple[Path, str]:
    if (policy_tag or "").strip():
        cands = list(raw_dir.glob(f"results_packets__{scenario}__ret{ret}__{policy_tag}__seed*.csv"))
    else:
        cands = list(raw_dir.glob(f"results_packets__{scenario}__ret{ret}__seed*.csv"))
        if not cands:
            cands = list(raw_dir.glob(f"results_packets__{scenario}__ret{ret}__*.csv"))
    if cands:
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        p = cands[0]
        tag = p.stem.split(f"__ret{ret}__")[-1]
        return p, tag

    p_old = raw_dir / f"results_packets__{scenario}__ret{ret}.csv"
    if p_old.exists():
        return p_old, "oldname"

    raise FileNotFoundError(
        f"Cannot find packets in {raw_dir}. "
        f"Expected results_packets__{scenario}__ret{ret}__seed*.csv (or oldname)."
    )




def _parse_band_edges(s: str) -> list[float]:
    vals = []
    for part in str(s or '').split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    vals = sorted(set(vals))
    return vals if len(vals) >= 2 else [0.0, 50.0, 100.0, 150.0, 200.0]


def _band_label(lo: float, hi: float) -> str:
    return f"{int(round(lo))}-{int(round(hi))}m"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=str, default="latest")
    ap.add_argument("--scenario", required=True, choices=["Ref", "UrbMask", "Tunnel"])
    ap.add_argument("--retrans", required=True, type=int, choices=[0, 1, 2])
    ap.add_argument("--policy_tag", type=str, default="")

    ap.add_argument("--profile", type=str, default="safety_awareness", choices=["safety_awareness", "pre_crash", "custom"])
    ap.add_argument("--deadline_ms", type=float, default=0.0)
    ap.add_argument("--grace_ms", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--msg_rate_hz", type=float, default=10.0)
    ap.add_argument("--prefer_packet_deadline", action="store_true")

    ap.add_argument("--dist_bin_m", type=float, default=5.0)
    ap.add_argument("--max_distance_m", type=float, default=200.0, help="Only keep packets with distance_m <= this value. Use <=0 to disable.")
    ap.add_argument("--band_min_m", type=float, default=80.0)
    ap.add_argument("--band_max_m", type=float, default=100.0)
    ap.add_argument("--mid_bin_m", type=float, default=20.0)
    ap.add_argument("--u_bin_w", type=float, default=0.05)
    ap.add_argument("--u_min", type=float, default=-0.25)
    ap.add_argument("--u_max", type=float, default=1.25)
    ap.add_argument("--band_edges_m", type=str, default="0,50,100,150,200")

    args = ap.parse_args()

    run_id = _pick_run_id(args.run_id)
    rp = ensure_run_dirs_a(run_id, save_as_latest=False, meta={"script": "analyze_tce_C.py"})

    log_command(run_id, rp.run_results_dir, extra=f"analyze_tce scenario={args.scenario} ret={args.retrans} profile={args.profile}")

    pkt_path, tag = _pick_latest_packets_file(rp.raw_dir, args.scenario, args.retrans, policy_tag=str(args.policy_tag))
    df = pd.read_csv(pkt_path)
    n_raw = int(len(df))
    if float(args.max_distance_m) > 0.0 and len(df) > 0 and "distance_m" in df.columns:
        dist = pd.to_numeric(df["distance_m"], errors="coerce")
        keep = dist.notna() & (dist <= float(args.max_distance_m))
        df = df.loc[keep].copy()
        print(f"[INFO] TCE distance filter <= {float(args.max_distance_m):.1f} m: kept {len(df)}/{n_raw} packets")

    cfg = resolve_tce_config(
        profile=args.profile,
        deadline_ms=float(args.deadline_ms),
        grace_ms=float(args.grace_ms),
        beta=float(args.beta),
        gamma=float(args.gamma),
        msg_rate_hz=float(args.msg_rate_hz),
    )
    if args.prefer_packet_deadline and ("deadline_ms" in df.columns):
        print("[INFO] TCE deadline policy: stricter(packet_deadline, profile_deadline)")
    if len(df) == 0:
        dfu = df.copy()
    else:
        dfu = compute_tce_utility(df, cfg=cfg, prefer_packet_deadline=bool(args.prefer_packet_deadline))

    profile_tag = cfg.profile
    if args.deadline_ms > 0:
        profile_tag += f"__D{int(round(cfg.deadline_ms))}"
    if args.grace_ms > 0:
        profile_tag += f"__G{int(round(cfg.grace_ms))}"

    overall = summarize_tce(dfu)
    overall_df = pd.DataFrame([{
        "scenario": args.scenario,
        "retrans": int(args.retrans),
        "policy_tag": str(dfu["policy_tag"].iloc[0]) if "policy_tag" in dfu.columns and len(dfu) > 0 else str(args.policy_tag),
        "retx_policy": str(dfu["retx_policy"].iloc[0]) if "retx_policy" in dfu.columns and len(dfu) > 0 else "classical",
        "profile": cfg.profile,
        "deadline_ms": float(cfg.deadline_ms),
        "grace_ms": float(cfg.grace_ms),
        "beta": float(cfg.beta),
        "gamma": float(cfg.gamma),
        "msg_rate_hz": float(cfg.msg_rate_hz),
        "analysis_max_distance_m": float(args.max_distance_m),
        **overall,
    }])
    overall_path = rp.tables_dir / f"tce_summary__{args.scenario}__ret{args.retrans}__{profile_tag}__{tag}.csv"
    overall_df.to_csv(overall_path, index=False)

    bin_w = float(args.dist_bin_m)
    dfu["dist_bin_left"] = (np.floor(pd.to_numeric(dfu["distance_m"], errors="coerce") / bin_w) * bin_w).astype(float)
    dfu["dist_bin_center"] = dfu["dist_bin_left"] + bin_w / 2.0
    bydist = aggregate_tce_by_group(dfu, "dist_bin_left")
    if len(bydist) > 0:
        bydist["dist_bin_center"] = pd.to_numeric(bydist["dist_bin_left"], errors="coerce") + bin_w / 2.0
        bydist = bydist.sort_values("dist_bin_left").reset_index(drop=True)
    bydist_path = rp.tables_dir / f"tce_by_distance__{args.scenario}__ret{args.retrans}__{profile_tag}__{tag}.csv"
    bydist.to_csv(bydist_path, index=False)

    band_edges = _parse_band_edges(args.band_edges_m)
    band_rows = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        band = dfu[(pd.to_numeric(dfu["distance_m"], errors="coerce") >= float(lo)) & (pd.to_numeric(dfu["distance_m"], errors="coerce") < float(hi))].copy()
        if len(band) == 0:
            continue
        s = summarize_tce(band)
        band_rows.append({
            "scenario": str(args.scenario),
            "retrans": int(args.retrans),
            "policy_tag": str(band["policy_tag"].iloc[0]) if "policy_tag" in band.columns else str(args.policy_tag),
            "retx_policy": str(band["retx_policy"].iloc[0]) if "retx_policy" in band.columns else "classical",
            "profile": str(cfg.profile),
            "band_label": _band_label(float(lo), float(hi)),
            "dist_lo_m": float(lo),
            "dist_hi_m": float(hi),
            **s,
        })
    band_path = rp.tables_dir / f"tce_by_band__{args.scenario}__ret{args.retrans}__{profile_tag}__{tag}.csv"
    pd.DataFrame(band_rows).to_csv(band_path, index=False)

    outputs = {"summary": overall_path.name, "by_distance": bydist_path.name, "by_band": band_path.name}

    util_cols = [
        "msg_id", "tx_time_s", "tx_id", "rx_id", "distance_m", "link_state",
        "success", "success_phy", "late", "fail_reason", "delay_ms", "deadline_ms",
        "deadline_eff_ms", "grace_eff_ms", "tardiness_ms", "utility_tce", "utility_late_partial",
        "link_bias", "hotspot_mult_col", "hotspot_mult_delay", "tx_speed_mps", "tx_road_tag",
    ]
    keep = [c for c in util_cols if c in dfu.columns]
    util_path = rp.tables_dir / f"tce_packets_compact__{args.scenario}__ret{args.retrans}__{profile_tag}__{tag}.csv"
    dfu[keep].to_csv(util_path, index=False)
    outputs["packets_compact"] = util_path.name

    if args.scenario == "UrbMask" and ("mid_x_m" in dfu.columns):
        band = dfu[
            (pd.to_numeric(dfu["distance_m"], errors="coerce") >= float(args.band_min_m)) &
            (pd.to_numeric(dfu["distance_m"], errors="coerce") < float(args.band_max_m))
        ].copy()
        band = band[np.isfinite(pd.to_numeric(band["mid_x_m"], errors="coerce"))].copy()
        if len(band) > 0:
            mid_bin = float(args.mid_bin_m)
            band["mid_x_bin_left"] = np.floor(pd.to_numeric(band["mid_x_m"], errors="coerce") / mid_bin) * mid_bin
            band["mid_x_bin_center"] = band["mid_x_bin_left"] + mid_bin / 2.0
            agg = aggregate_tce_by_group(band, "mid_x_bin_left")
            if len(agg) > 0:
                agg["mid_x_bin_center"] = pd.to_numeric(agg["mid_x_bin_left"], errors="coerce") + mid_bin / 2.0
                agg = agg.sort_values("mid_x_bin_left").reset_index(drop=True)
            f4 = rp.tables_dir / f"tce_position_heterogeneity__UrbMask__ret{args.retrans}__{profile_tag}__band{int(args.band_min_m)}-{int(args.band_max_m)}__{tag}.csv"
            agg.to_csv(f4, index=False)
            outputs["UrbMask_F4_TCE"] = f4.name

    if args.scenario == "Tunnel" and ("tunnel_u" in dfu.columns):
        band = dfu[
            (pd.to_numeric(dfu["distance_m"], errors="coerce") >= float(args.band_min_m)) &
            (pd.to_numeric(dfu["distance_m"], errors="coerce") < float(args.band_max_m))
        ].copy()
        band = band[np.isfinite(pd.to_numeric(band["tunnel_u"], errors="coerce"))].copy()
        if len(band) > 0:
            u_bin = float(args.u_bin_w)
            uu = pd.to_numeric(band["tunnel_u"], errors="coerce")
            band = band[(uu >= float(args.u_min)) & (uu < float(args.u_max))].copy()
            if len(band) > 0:
                band["u_bin_left"] = np.floor(pd.to_numeric(band["tunnel_u"], errors="coerce") / u_bin) * u_bin
                band["u_bin_center"] = band["u_bin_left"] + u_bin / 2.0
                agg = aggregate_tce_by_group(band, "u_bin_left")
                if len(agg) > 0:
                    agg["u_bin_center"] = pd.to_numeric(agg["u_bin_left"], errors="coerce") + u_bin / 2.0
                    agg = agg.sort_values("u_bin_left").reset_index(drop=True)
                f5 = rp.tables_dir / f"tce_tunnel_segments__Tunnel__ret{args.retrans}__{profile_tag}__band{int(args.band_min_m)}-{int(args.band_max_m)}__{tag}.csv"
                agg.to_csv(f5, index=False)
                outputs["Tunnel_F5_TCE"] = f5.name

    update_manifest(
        rp.manifest_path,
        {
            "last_tce_analyze": {
                "scenario": args.scenario,
                "retrans": int(args.retrans),
                "policy_tag": str(args.policy_tag),
                "profile": cfg.profile,
                "deadline_ms": float(cfg.deadline_ms),
                "grace_ms": float(cfg.grace_ms),
                "beta": float(cfg.beta),
                "gamma": float(cfg.gamma),
                "msg_rate_hz": float(cfg.msg_rate_hz),
                "prefer_packet_deadline": bool(args.prefer_packet_deadline),
                "packets_file": pkt_path.name,
                "outputs": outputs,
            }
        },
    )

    print(f"[OK] run_id={run_id}")
    print(f"[OK] packets -> {pkt_path.name}")
    if args.policy_tag:
        print(f"[INFO] policy_tag = {args.policy_tag}")
    print(f"[INFO] overall timely_success_rate = {overall['timely_success_rate']:.6f}")
    print(f"[INFO] overall phy_success_rate    = {overall['phy_success_rate']:.6f}")
    print(f"[INFO] overall tce                 = {overall['tce']:.6f}")
    print(f"[INFO] overall n_late             = {overall['n_late']}")
    print(f"[INFO] overall late_partial_gain  = {overall['late_partial_gain']:.6f}")

    if overall["n_late"] == 0:
        print("[WARN] No late-but-received packets under the current analysis deadline.")
        if args.prefer_packet_deadline:
            print("[HINT] --prefer_packet_deadline is active. The effective deadline is the stricter of the packet deadline and the profile deadline.")

    print(f"[OK] tce_summary -> {overall_path.name}")
    print(f"[OK] tce_by_distance -> {bydist_path.name}")

    if len(bydist) > 0:
        show_cols = ["dist_bin_center", "phy_success_rate", "tce", "timely_success_rate", "n_late", "late_partial_gain"]
        show_cols = [c for c in show_cols if c in bydist.columns]
        print("[INFO] first bins:")
        print(bydist[show_cols].head(12).to_string(index=False))

    if "UrbMask_F4_TCE" in outputs:
        print(f"[OK] TCE F4 -> {outputs['UrbMask_F4_TCE']}")
    if "Tunnel_F5_TCE" in outputs:
        print(f"[OK] TCE F5 -> {outputs['Tunnel_F5_TCE']}")


if __name__ == "__main__":
    main()