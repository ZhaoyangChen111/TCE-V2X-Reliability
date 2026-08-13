from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def ecdf(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def _pick_latest_summary_file(tables_dir: Path, scenario: str, ret: int, policy_tag: str = "") -> tuple[Path, str]:
    if (policy_tag or "").strip():
        cands = list(tables_dir.glob(f"summary_metrics__{scenario}__ret{ret}__{policy_tag}__*.csv"))
    else:
        cands = list(tables_dir.glob(f"summary_metrics__{scenario}__ret{ret}__seed*.csv"))
        if not cands:
            cands = list(tables_dir.glob(f"summary_metrics__{scenario}__ret{ret}__*.csv"))
    if not cands:
        raise FileNotFoundError(f"summary not found in {tables_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    p = cands[0]
    tag = p.stem.split(f"__ret{ret}__")[-1]
    return p, tag


def _pick_latest_packets_file(raw_dir: Path, scenario: str, ret: int, policy_tag: str = "") -> Path:
    # allow .csv and .csv.gz in case you later compress raw
    if (policy_tag or "").strip():
        cands = list(raw_dir.glob(f"results_packets__{scenario}__ret{ret}__{policy_tag}__seed*.csv*"))
    else:
        cands = list(raw_dir.glob(f"results_packets__{scenario}__ret{ret}__*.csv*"))
    if cands:
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0]
    p_old = raw_dir / f"results_packets__{scenario}__ret{ret}.csv"
    if p_old.exists():
        return p_old
    raise FileNotFoundError(f"packets not found in {raw_dir}")


def _smooth_by_distance(x: np.ndarray, y: np.ndarray, window_m: float) -> np.ndarray:
    """
    Rolling mean smoothing in distance domain.
    NOTE: y may contain NaN; rolling mean ignores NaN by default.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 5 or window_m <= 1e-9:
        return y

    dx = np.diff(x)
    dx = dx[np.isfinite(dx)]
    if len(dx) == 0:
        return y

    step = float(np.median(dx))
    if step <= 1e-9:
        return y

    win = int(round(float(window_m) / step))
    win = max(3, win)
    if win % 2 == 0:
        win += 1

    ys = pd.Series(y).rolling(window=win, center=True, min_periods=max(1, win // 3)).mean()
    return ys.to_numpy(dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=str, default="latest")
    ap.add_argument("--scenario", required=True, choices=["Ref", "UrbMask", "Tunnel"])
    ap.add_argument("--retrans", type=int, required=True)
    ap.add_argument("--policy_tag", type=str, default="")

    # F1/F3 distance-axis cap
    ap.add_argument("--x_max_m", type=float, default=200.0, help="Default 200 m for smoke/paper-style diagnostic plots. Use 0 for full-range output.")

    # filter bins by sample count (recommended for clean paper plots)
    ap.add_argument(
        "--min_bin_count",
        type=int,
        default=0,
        help="0 => no filter; else drop distance bins with n_total < min_bin_count for F1/F3",
    )

    # F2 (CDF) distance filter (default OFF)
    ap.add_argument("--cdf_max_dist_m", type=float, default=200.0, help="Default 200 m for smoke/paper-style diagnostic plots. Use 0 for full-range output.")

    # NEW: plotting style for F1/F3
    ap.add_argument(
        "--curve_style",
        type=str,
        default="line",
        choices=["line", "points", "smooth"],
        help="line: no markers; points: marker per bin; smooth: rolling-mean curve (optionally overlay raw)",
    )
    ap.add_argument("--smooth_window_m", type=float, default=40.0, help="Used when curve_style=smooth (meters).")
    ap.add_argument("--smooth_overlay_raw", action="store_true", help="If set, smooth plots also overlay raw line lightly.")

    args = ap.parse_args()

    run_id = _pick_run_id(args.run_id)
    rp = ensure_run_dirs_a(run_id, save_as_latest=False, meta={"script": "plot_figures_C.py"})

    log_command(run_id, rp.run_results_dir, extra=f"plot scenario={args.scenario} ret={args.retrans}")
    update_manifest(
        rp.manifest_path,
        {
            "last_plot": {
                "scenario": args.scenario,
                "retrans": int(args.retrans),
                "policy_tag": str(args.policy_tag),
                "x_max_m": float(args.x_max_m),
                "cdf_max_dist_m": float(args.cdf_max_dist_m),
                "min_bin_count": int(args.min_bin_count),
                "curve_style": str(args.curve_style),
                "smooth_window_m": float(args.smooth_window_m),
                "smooth_overlay_raw": bool(args.smooth_overlay_raw),
            }
        },
    )

    scenario = args.scenario
    ret = int(args.retrans)

    summ_path, tag = _pick_latest_summary_file(rp.tables_dir, scenario, ret, policy_tag=str(args.policy_tag))
    pkt_path = _pick_latest_packets_file(rp.raw_dir, scenario, ret, policy_tag=str(args.policy_tag))

    summ = pd.read_csv(summ_path).sort_values("dist_bin_center")
    pkt = pd.read_csv(pkt_path)  # supports .csv.gz by infer

    # ---- F1/F3 only: apply x_max_m + min_bin_count to summary ----
    xcap = float(args.x_max_m)
    summ_plot = summ
    if xcap > 1e-9:
        summ_plot = summ_plot[summ_plot["dist_bin_center"] <= xcap].copy()

    minc = int(args.min_bin_count)
    if minc > 0 and ("n_total" in summ_plot.columns):
        summ_plot = summ_plot[summ_plot["n_total"] >= minc].copy()

    x = summ_plot["dist_bin_center"].to_numpy(dtype=float)
    y_pdr = summ_plot["pdr"].to_numpy(dtype=float)
    y95 = summ_plot["delay_p95_ms"].to_numpy(dtype=float) if "delay_p95_ms" in summ_plot.columns else np.full_like(x, np.nan)
    y99 = summ_plot["delay_p99_ms"].to_numpy(dtype=float) if "delay_p99_ms" in summ_plot.columns else np.full_like(x, np.nan)

    style = str(args.curve_style).lower()
    if style == "smooth":
        y_pdr_s = _smooth_by_distance(x, y_pdr, float(args.smooth_window_m))
        y95_s = _smooth_by_distance(x, y95, float(args.smooth_window_m))
        y99_s = _smooth_by_distance(x, y99, float(args.smooth_window_m))
    else:
        y_pdr_s, y95_s, y99_s = y_pdr, y95, y99

    outputs: list[Path] = []

    # ----------------- F1 -----------------
    fig1 = rp.figures_dir / f"F1_PDR_vs_distance__{scenario}__ret{ret}__{tag}.png"
    plt.figure()

    if style == "points":
        plt.plot(x, y_pdr, marker="o")
    elif style == "smooth":
        if args.smooth_overlay_raw:
            plt.plot(x, y_pdr, alpha=0.35)
        plt.plot(x, y_pdr_s)
    else:
        plt.plot(x, y_pdr)

    plt.xlabel("Distance (m)")
    plt.ylabel("PDR")
    plt.title(f"{scenario}: PDR vs Distance (ret={ret})" if not args.policy_tag else f"{scenario}: PDR vs Distance (ret={ret}, {args.policy_tag})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(-0.05, 1.05)
    if xcap > 1e-9:
        plt.xlim(0.0, xcap)
    plt.tight_layout()
    plt.savefig(fig1, dpi=220)
    plt.close()
    outputs.append(fig1)

    # ----------------- F3 -----------------
    fig3 = rp.figures_dir / f"F3_Delay_p95_p99_vs_distance__{scenario}__ret{ret}__{tag}.png"
    plt.figure()

    if style == "points":
        plt.plot(x, y95, marker="o", label="p95")
        plt.plot(x, y99, marker="o", label="p99")
    elif style == "smooth":
        if args.smooth_overlay_raw:
            plt.plot(x, y95, alpha=0.35, label="p95(raw)")
            plt.plot(x, y99, alpha=0.35, label="p99(raw)")
        plt.plot(x, y95_s, label="p95")
        plt.plot(x, y99_s, label="p99")
    else:
        plt.plot(x, y95, label="p95")
        plt.plot(x, y99, label="p99")

    plt.xlabel("Distance (m)")
    plt.ylabel("Delay (ms)")
    plt.title(f"{scenario}: Delay p95/p99 vs Distance (ret={ret})" if not args.policy_tag else f"{scenario}: Delay p95/p99 vs Distance (ret={ret}, {args.policy_tag})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    if xcap > 1e-9:
        plt.xlim(0.0, xcap)
    plt.tight_layout()
    plt.savefig(fig3, dpi=220)
    plt.close()
    outputs.append(fig3)

    # ----------------- F2 (CDF) -----------------
    cdf_dmax = float(args.cdf_max_dist_m)
    pkt_cdf = pkt
    if cdf_dmax > 1e-9 and ("distance_m" in pkt_cdf.columns):
        pkt_cdf = pkt_cdf[pkt_cdf["distance_m"] <= cdf_dmax].copy()

    delays = pkt_cdf.loc[pkt_cdf["success"] == 1, "delay_ms"].to_numpy(dtype=float)
    x2, y2 = ecdf(delays)

    tag_f2 = tag
    if ("__dmax" in tag) and (cdf_dmax <= 1e-9):
        tag_f2 = tag.split("__dmax")[0] + "__cdfAll"

    fig2 = rp.figures_dir / f"F2_Delay_CDF__{scenario}__ret{ret}__{tag_f2}.png"
    plt.figure()
    if len(x2) > 0:
        plt.plot(x2, y2)
    plt.xlabel("Delay (ms)")
    plt.ylabel("CDF")
    ttl = f"{scenario}: Delay CDF (ret={ret})" if not args.policy_tag else f"{scenario}: Delay CDF (ret={ret}, {args.policy_tag})"
    if cdf_dmax > 1e-9:
        ttl += f", dist<={int(cdf_dmax)}m"
    plt.title(ttl)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig2, dpi=220)
    plt.close()
    outputs.append(fig2)

    update_manifest(rp.manifest_path, {"last_plot_outputs": [p.name for p in outputs]})

    print(f"[OK] run_id={run_id}")
    print("[OK] figures ->")
    for pp in outputs:
        if pp.exists():
            print(" -", str(pp))


if __name__ == "__main__":
    main()