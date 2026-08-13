from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paths_C import ensure_run_dirs_a, make_run_id, load_latest_run_id, default_run_prefix
from run_logging import log_command, update_manifest


def _pick_run_id(arg_run_id: str) -> str:
    s = (arg_run_id or "").strip()
    if s == "":
        return make_run_id(prefix=default_run_prefix())
    if s.lower() == "latest":
        rid = load_latest_run_id()
        return rid if rid else make_run_id(prefix=default_run_prefix())
    return s


def _quantiles_ms(x: pd.Series) -> tuple[float, float, float]:
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    q = np.quantile(x.to_numpy(dtype=float), [0.50, 0.95, 0.99], method="linear")
    return (float(q[0]), float(q[1]), float(q[2]))


def _pick_latest_packets_file(raw_dir: Path, scenario: str, ret: int, policy_tag: str = "") -> tuple[Path, str]:
    """
    Prefer new naming:
      results_packets__{scenario}__ret{ret}__seed*.csv
    If multiple exist (reruns), pick the latest by mtime.
    Return (path, tag), where tag is used for output filenames.
    """
    if (policy_tag or "").strip():
        cands = list(raw_dir.glob(f"results_packets__{scenario}__ret{ret}__{policy_tag}__seed*.csv"))
    else:
        cands = list(raw_dir.glob(f"results_packets__{scenario}__ret{ret}__seed*.csv"))
        if not cands:
            cands = list(raw_dir.glob(f"results_packets__{scenario}__ret{ret}__*.csv"))
    if cands:
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        p = cands[0]
        tag = p.stem.split(f"__ret{ret}__")[-1]  # seed1-3 / seed5 etc
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

    # distance-bin for F1/F3
    ap.add_argument("--dist_bin_m", type=float, default=5.0)
    ap.add_argument("--max_distance_m", type=float, default=200.0, help="Only keep packets with distance_m <= this value. Use <=0 to disable.")
    ap.add_argument("--min_total_per_bin", type=int, default=500)
    ap.add_argument("--min_success_per_bin", type=int, default=50)

    ap.add_argument("--emit_phy_mechanism_tables", action="store_true")

    # F4/F5 common band (distance)
    ap.add_argument("--band_min_m", type=float, default=80.0)
    ap.add_argument("--band_max_m", type=float, default=100.0)

    # F4 (UrbMask heterogeneity): mid-x bin width
    ap.add_argument("--mid_bin_m", type=float, default=20.0)

    # F5 (Tunnel segments): u binning window
    ap.add_argument("--u_bin_w", type=float, default=0.05)
    ap.add_argument("--u_min", type=float, default=-0.25)
    ap.add_argument("--u_max", type=float, default=1.25)
    ap.add_argument("--band_edges_m", type=str, default="0,50,100,150,200")

    args = ap.parse_args()

    run_id = _pick_run_id(args.run_id)
    rp = ensure_run_dirs_a(run_id, save_as_latest=False, meta={"script": "analyze_metrics_C.py"})

    log_command(run_id, rp.run_results_dir, extra=f"analyze scenario={args.scenario} ret={args.retrans}")
    update_manifest(
        rp.manifest_path,
        {
            "last_analyze": {
                "scenario": args.scenario,
                "retrans": int(args.retrans),
                "policy_tag": str(args.policy_tag),
                "dist_bin_m": float(args.dist_bin_m),
                "max_distance_m": float(args.max_distance_m),
                "min_total_per_bin": int(args.min_total_per_bin),
                "min_success_per_bin": int(args.min_success_per_bin),
                "band_min_m": float(args.band_min_m),
                "band_max_m": float(args.band_max_m),
                "mid_bin_m": float(args.mid_bin_m),
                "u_bin_w": float(args.u_bin_w),
                "u_min": float(args.u_min),
                "u_max": float(args.u_max),
            }
        },
    )

    # load packets (latest only)
    pkt_path, tag = _pick_latest_packets_file(rp.raw_dir, args.scenario, args.retrans, policy_tag=str(args.policy_tag))
    df = pd.read_csv(pkt_path)
    n_raw = int(len(df))
    if float(args.max_distance_m) > 0.0 and len(df) > 0 and "distance_m" in df.columns:
        dist = pd.to_numeric(df["distance_m"], errors="coerce")
        keep = dist.notna() & (dist <= float(args.max_distance_m))
        df = df.loc[keep].copy()
        print(f"[INFO] analysis distance filter <= {float(args.max_distance_m):.1f} m: kept {len(df)}/{n_raw} packets")

    update_manifest(rp.manifest_path, {"last_analyze_inputs": {"packets_file": str(pkt_path.name), "packets_tag": tag}})

    if len(df) == 0:
        empty_cols = [
            "policy_tag", "retx_policy", "dist_bin_left", "dist_bin_center", "n_total", "n_success", "pdr",
            "n_success_phy", "pdr_phy", "n_late", "late_ratio_phy", "delay_p50_ms", "delay_p95_ms", "delay_p99_ms",
            "nlos_ratio", "tunnel_ratio", "avg_blockage_b", "avg_n_cs", "avg_cbr", "avg_p_col", "avg_cong_delay_ms",
        ]
        out = pd.DataFrame(columns=empty_cols)
        out_path = rp.tables_dir / f"summary_metrics__{args.scenario}__ret{args.retrans}__{tag}.csv"
        out.to_csv(out_path, index=False)
        print(f"[OK] run_id={run_id}")
        print(f"[OK] packets -> {pkt_path.name}")
        print(f"[WARN] packets file is empty; emitted empty summary -> {out_path.name}")
        update_manifest(rp.manifest_path, {"last_analyze_outputs": {"summary": out_path.name}})
        return

    # -------- summary_metrics (F1/F3) --------
    bin_w = float(args.dist_bin_m)
    df["dist_bin_left"] = (np.floor(df["distance_m"] / bin_w) * bin_w).astype(int)
    df["dist_bin_center"] = df["dist_bin_left"].astype(float) + bin_w / 2.0

    has_phy = "success_phy" in df.columns
    has_late = "late" in df.columns
    has_cbr = "cbr" in df.columns

    rows = []
    warn_total = 0
    warn_succ = 0

    for dist_left, g in df.groupby("dist_bin_left", sort=True):
        n_total = int(len(g))
        n_success = int(g["success"].sum())  # timely success

        pdr = float(n_success / n_total) if n_total > 0 else np.nan
        if n_total < int(args.min_total_per_bin):
            pdr = np.nan
            warn_total += 1

        succ_delays = g.loc[g["success"] == 1, "delay_ms"].dropna()
        if n_success < int(args.min_success_per_bin):
            d50, d95, d99 = (np.nan, np.nan, np.nan)
            warn_succ += 1
        else:
            d50, d95, d99 = _quantiles_ms(succ_delays)

        # physical success / late ratio (optional)
        if has_phy:
            n_success_phy = int(g["success_phy"].sum())
            pdr_phy = float(n_success_phy / n_total) if n_total > 0 else np.nan
        else:
            n_success_phy = n_success
            pdr_phy = float(n_success / n_total) if n_total > 0 else np.nan

        if has_late:
            n_late = int(g["late"].sum())
            late_ratio_phy = float(n_late / max(1, n_success_phy))
        else:
            n_late = 0
            late_ratio_phy = np.nan

        rows.append(
            {
                "scenario": str(args.scenario),
                "retrans": int(args.retrans),
                "policy_tag": str(g["policy_tag"].iloc[0]) if "policy_tag" in g.columns and len(g) > 0 else str(args.policy_tag),
                "retx_policy": str(g["retx_policy"].iloc[0]) if "retx_policy" in g.columns and len(g) > 0 else "classical",
                "dist_bin_left": int(dist_left),
                "dist_bin_center": float(dist_left) + bin_w / 2.0,
                "n_total": n_total,
                "n_success": n_success,
                "pdr": float(pdr),
                "n_success_phy": int(n_success_phy),
                "pdr_phy": float(pdr_phy),
                "n_late": int(n_late),
                "late_ratio_phy": float(late_ratio_phy),
                "delay_p50_ms": float(d50),
                "delay_p95_ms": float(d95),
                "delay_p99_ms": float(d99),
                "nlos_ratio": float((g["link_state"] == "NLOS").mean()) if "link_state" in g.columns else np.nan,
                "tunnel_ratio": float((g["link_state"] == "TUNNEL").mean()) if "link_state" in g.columns else np.nan,
                "avg_blockage_b": float(g["blockage_b"].mean()) if "blockage_b" in g.columns else np.nan,
                "avg_n_cs": float(g["n_cs"].mean()) if "n_cs" in g.columns else np.nan,
                "avg_cbr": float(g["cbr"].mean()) if has_cbr else np.nan,
                "avg_p_col": float(g["p_col"].mean()) if "p_col" in g.columns else np.nan,
                "avg_cong_delay_ms": float(g["cong_delay_ms"].mean()) if "cong_delay_ms" in g.columns else np.nan,
            }
        )

    out = pd.DataFrame(rows).sort_values("dist_bin_center").reset_index(drop=True)
    out_path = rp.tables_dir / f"summary_metrics__{args.scenario}__ret{args.retrans}__{tag}.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] run_id={run_id}")
    print(f"[OK] packets -> {pkt_path.name}")
    if args.policy_tag:
        print(f"[INFO] policy_tag = {args.policy_tag}")
    print(f"[OK] summary -> {out_path} (rows={len(out)})")
    if warn_total:
        print(f"[warn] dist bins with n_total<{args.min_total_per_bin}: {warn_total} (PDR->NaN)")
    if warn_succ:
        print(f"[warn] dist bins with n_success<{args.min_success_per_bin}: {warn_succ} (delay quantiles->NaN)")

    # -------- distance-band summary (helps keep overall numbers interpretable) --------
    band_edges = _parse_band_edges(args.band_edges_m)
    band_rows = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        band = df[(pd.to_numeric(df["distance_m"], errors="coerce") >= float(lo)) & (pd.to_numeric(df["distance_m"], errors="coerce") < float(hi))].copy()
        if len(band) == 0:
            continue
        succ = pd.to_numeric(band["success"], errors="coerce").fillna(0)
        succ_phy = pd.to_numeric(band.get("success_phy", succ), errors="coerce").fillna(0)
        late = pd.to_numeric(band.get("late", 0), errors="coerce").fillna(0)
        dsucc = pd.to_numeric(band.get("delay_ms", np.nan), errors="coerce")
        dsucc = dsucc[succ > 0.5]
        d50, d95, d99 = _quantiles_ms(dsucc.dropna())
        band_rows.append({
            "scenario": str(args.scenario),
            "retrans": int(args.retrans),
            "policy_tag": str(band["policy_tag"].iloc[0]) if "policy_tag" in band.columns else str(args.policy_tag),
            "retx_policy": str(band["retx_policy"].iloc[0]) if "retx_policy" in band.columns else "classical",
            "band_label": _band_label(float(lo), float(hi)),
            "dist_lo_m": float(lo),
            "dist_hi_m": float(hi),
            "n_total": int(len(band)),
            "timely_rate": float(succ.mean()),
            "phy_rate": float(succ_phy.mean()),
            "late_ratio_total": float(late.mean()),
            "late_ratio_phy": float(late.sum() / max(1, succ_phy.sum())),
            "avg_attempts": float(pd.to_numeric(band["n_tx_attempts"], errors="coerce").mean()),
            "delay_p50_ms": float(d50),
            "delay_p95_ms": float(d95),
            "delay_p99_ms": float(d99),
            "avg_cbr": float(pd.to_numeric(band.get("cbr", np.nan), errors="coerce").mean()),
            "avg_p_col": float(pd.to_numeric(band.get("p_col", np.nan), errors="coerce").mean()),
            "avg_cong_delay_ms": float(pd.to_numeric(band.get("cong_delay_ms", np.nan), errors="coerce").mean()),
        })
    band_path = rp.tables_dir / f"summary_metrics_bands__{args.scenario}__ret{args.retrans}__{tag}.csv"
    pd.DataFrame(band_rows).to_csv(band_path, index=False)

    # -------- UrbMask legacy PHY-only mechanism table (optional) --------
    if args.scenario == "UrbMask" and args.emit_phy_mechanism_tables:
        band_min = float(args.band_min_m)
        band_max = float(args.band_max_m)
        band = df[(df["distance_m"] >= band_min) & (df["distance_m"] < band_max)].copy()
        if len(band) == 0:
            print("[info] F4 skipped: empty band")
            update_manifest(rp.manifest_path, {"last_analyze_outputs": {"summary": out_path.name, "F4": None}})
            return

        # Prefer mid_x_m directly (new sim output). If absent, skip F4 (no expensive traj join here).
        if "mid_x_m" not in band.columns:
            print("[info] F4 skipped: packets missing mid_x_m")
            update_manifest(rp.manifest_path, {"last_analyze_outputs": {"summary": out_path.name, "F4": None}})
            return

        mid_bin = float(args.mid_bin_m)
        band = band[np.isfinite(band["mid_x_m"])].copy()
        band["mid_x_bin_left"] = np.floor(band["mid_x_m"] / mid_bin) * mid_bin
        band["mid_x_bin_center"] = band["mid_x_bin_left"] + mid_bin / 2.0

        agg = band.groupby("mid_x_bin_left", dropna=True).agg(
            mid_x_bin_center=("mid_x_bin_center", "first"),
            n_total=("success", "size"),
            n_success=("success", "sum"),
        ).reset_index()

        agg["pdr_band"] = agg["n_success"] / agg["n_total"]

        if "success_phy" in band.columns:
            tmp = band.groupby("mid_x_bin_left")["success_phy"].mean().reset_index().rename(columns={"success_phy": "pdr_phy_band"})
            agg = agg.merge(tmp, on="mid_x_bin_left", how="left")
        else:
            agg["pdr_phy_band"] = np.nan

        if "late" in band.columns and "success_phy" in band.columns:
            tmp2 = band.groupby("mid_x_bin_left").apply(
                lambda gg: float(gg["late"].sum() / max(1, gg["success_phy"].sum()))
            ).reset_index(name="late_ratio_phy_band")
            agg = agg.merge(tmp2, on="mid_x_bin_left", how="left")
        else:
            agg["late_ratio_phy_band"] = np.nan

        if "link_state" in band.columns:
            tmp3 = band.groupby("mid_x_bin_left")["link_state"].apply(lambda s: float((s == "NLOS").mean())).reset_index()
            tmp3 = tmp3.rename(columns={"link_state": "nlos_ratio_band"})
            agg = agg.merge(tmp3, on="mid_x_bin_left", how="left")
        else:
            agg["nlos_ratio_band"] = np.nan

        if "cbr" in band.columns:
            tmp4 = band.groupby("mid_x_bin_left")["cbr"].mean().reset_index().rename(columns={"cbr": "avg_cbr_band"})
            agg = agg.merge(tmp4, on="mid_x_bin_left", how="left")
        else:
            agg["avg_cbr_band"] = np.nan

        agg = agg.sort_values("mid_x_bin_left").reset_index(drop=True)
        f4_path = rp.tables_dir / f"position_heterogeneity__UrbMask__ret{args.retrans}__band{int(band_min)}-{int(band_max)}__{tag}.csv"
        agg.to_csv(f4_path, index=False)
        print(f"[OK] F4 table -> {f4_path} (rows={len(agg)})")

        update_manifest(rp.manifest_path, {"last_analyze_outputs": {"summary": out_path.name, "F4": f4_path.name}})
        return

    # -------- Tunnel legacy PHY-only mechanism table (optional) --------
    if args.scenario == "Tunnel" and args.emit_phy_mechanism_tables:
        for col in ["tunnel_u", "distance_m", "success", "delay_ms"]:
            if col not in df.columns:
                print(f"[info] F5 skipped: missing {col}")
                update_manifest(rp.manifest_path, {"last_analyze_outputs": {"summary": out_path.name, "F5": None}})
                return

        band_min = float(args.band_min_m)
        band_max = float(args.band_max_m)
        u_min = float(args.u_min)
        u_max = float(args.u_max)
        u_bin_w = float(args.u_bin_w)

        band = df[(df["distance_m"] >= band_min) & (df["distance_m"] < band_max)].copy()
        band = band[np.isfinite(band["tunnel_u"])].copy()
        if len(band) == 0:
            print("[info] F5 skipped: empty band or tunnel_u all NaN")
            update_manifest(rp.manifest_path, {"last_analyze_outputs": {"summary": out_path.name, "F5": None}})
            return

        band = band[(band["tunnel_u"] >= u_min) & (band["tunnel_u"] <= u_max)].copy()
        if len(band) == 0:
            print("[info] F5 skipped: empty after u-window filter")
            update_manifest(rp.manifest_path, {"last_analyze_outputs": {"summary": out_path.name, "F5": None}})
            return

        band["u_bin_left"] = np.floor(band["tunnel_u"] / u_bin_w) * u_bin_w
        band["u_bin_center"] = band["u_bin_left"] + u_bin_w / 2.0

        rows2 = []
        for u_left, g2 in band.groupby("u_bin_left", sort=True):
            n_total = int(len(g2))
            n_success = int(g2["success"].sum())

            pdr_band = float(n_success / n_total) if n_total > 0 else np.nan

            if "success_phy" in g2.columns:
                pdr_phy_band = float(g2["success_phy"].mean())
            else:
                pdr_phy_band = np.nan

            if "late" in g2.columns and "success_phy" in g2.columns:
                late_ratio_phy_band = float(g2["late"].sum() / max(1, g2["success_phy"].sum()))
            else:
                late_ratio_phy_band = np.nan

            succ_delays = g2.loc[g2["success"] == 1, "delay_ms"].dropna()
            if n_success < int(args.min_success_per_bin):
                p95 = np.nan
                p99 = np.nan
            else:
                _, p95, p99 = _quantiles_ms(succ_delays)

            rows2.append(
                {
                    "u_bin_left": float(u_left),
                    "u_bin_center": float(u_left + u_bin_w / 2.0),
                    "n_total": n_total,
                    "n_success": n_success,
                    "pdr_band": float(pdr_band),
                    "pdr_phy_band": float(pdr_phy_band),
                    "late_ratio_phy_band": float(late_ratio_phy_band),
                    "delay_p95_band_ms": float(p95),
                    "delay_p99_band_ms": float(p99),
                    "mean_blockage_b": float(g2["blockage_b"].mean()) if "blockage_b" in g2.columns else np.nan,
                }
            )

        seg = pd.DataFrame(rows2).sort_values("u_bin_left").reset_index(drop=True)
        f5_path = rp.tables_dir / f"tunnel_segments__Tunnel__ret{args.retrans}__band{int(band_min)}-{int(band_max)}__{tag}.csv"
        seg.to_csv(f5_path, index=False)
        print(f"[OK] F5 table -> {f5_path} (rows={len(seg)})")

        update_manifest(rp.manifest_path, {"last_analyze_outputs": {"summary": out_path.name, "F5": f5_path.name}})
        return

    # Ref: only summary
    update_manifest(rp.manifest_path, {"last_analyze_outputs": {"summary": out_path.name}})


if __name__ == "__main__":
    main()