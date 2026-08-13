from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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


def _pick_latest_tce_distance_file(tables_dir: Path, scenario: str, ret: int, profile: str, policy_tag: str = "") -> tuple[Path, str]:
    if (policy_tag or "").strip():
        cands = list(tables_dir.glob(f"tce_by_distance__{scenario}__ret{ret}__{profile}*__{policy_tag}__*.csv"))
        if not cands:
            cands = list(tables_dir.glob(f"tce_by_distance__{scenario}__ret{ret}__{profile}*__{policy_tag}*.csv"))
    else:
        cands = list(tables_dir.glob(f"tce_by_distance__{scenario}__ret{ret}__{profile}*__seed*.csv"))
        if not cands:
            cands = list(tables_dir.glob(f"tce_by_distance__{scenario}__ret{ret}__{profile}*__*.csv"))
    if not cands:
        raise FileNotFoundError(
            f"TCE by-distance table not found in {tables_dir} "
            f"for scenario={scenario} ret={ret} profile={profile}"
        )
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    p = cands[0]
    tag = p.stem.split(f"__{profile}")[-1].lstrip("_")
    return p, tag


def _safe_series(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise KeyError(f"Required column not found: {col}")
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _nanmax_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    return float(np.max(np.abs(a[m] - b[m])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=str, default="latest")
    ap.add_argument("--scenario", required=True, choices=["Ref", "UrbMask", "Tunnel"])
    ap.add_argument("--retrans", type=int, required=True, choices=[0, 1, 2])
    ap.add_argument("--policy_tag", type=str, default="")
    ap.add_argument("--profile", type=str, default="safety_awareness")
    ap.add_argument("--x_max_m", type=float, default=0.0)
    ap.add_argument("--min_bin_count", type=int, default=0)
    ap.add_argument("--y_max", type=float, default=0.0)
    ap.add_argument("--zoom_y_max", type=float, default=0.12)
    ap.add_argument("--zoom_x_max_m", type=float, default=200.0)
    ap.add_argument("--save_zoom", action="store_true")
    args = ap.parse_args()

    run_id = _pick_run_id(args.run_id)
    rp = ensure_run_dirs_a(run_id, save_as_latest=False, meta={"script": "plot_tce_C.py"})

    log_command(
        run_id,
        rp.run_results_dir,
        extra=f"plot_tce scenario={args.scenario} ret={args.retrans} profile={args.profile}"
    )

    src, tag = _pick_latest_tce_distance_file(rp.tables_dir, args.scenario, args.retrans, args.profile, policy_tag=str(args.policy_tag))
    df = pd.read_csv(src).sort_values("dist_bin_left").reset_index(drop=True)

    if args.min_bin_count > 0 and "n_total" in df.columns:
        df = df[df["n_total"] >= int(args.min_bin_count)].copy()

    if args.x_max_m > 0 and "dist_bin_center" in df.columns:
        df = df[df["dist_bin_center"] <= float(args.x_max_m)].copy()

    if len(df) == 0:
        raise RuntimeError("No rows remain after filtering.")

    x = _safe_series(df, "dist_bin_center")
    y_tce = _safe_series(df, "tce")
    y_timely = _safe_series(df, "timely_success_rate")
    y_phy = _safe_series(df, "phy_success_rate")

    gap_phy_tce = _nanmax_abs_diff(y_phy, y_tce)
    gap_tce_timely = _nanmax_abs_diff(y_tce, y_timely)
    gap_phy_timely = _nanmax_abs_diff(y_phy, y_timely)

    print(f"[INFO] source_table = {src.name}")
    print(f"[INFO] max|PHY - TCE|      = {gap_phy_tce:.6f}")
    print(f"[INFO] max|TCE - Timely|   = {gap_tce_timely:.6f}")
    print(f"[INFO] max|PHY - Timely|   = {gap_phy_timely:.6f}")

    finite_vals = np.concatenate([
        y_phy[np.isfinite(y_phy)],
        y_tce[np.isfinite(y_tce)],
        y_timely[np.isfinite(y_timely)],
    ])
    data_ymax = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0

    if args.y_max > 0:
        y_top = float(args.y_max)
    else:
        y_top = max(0.05, min(1.05, data_ymax * 1.10))

    mark_every = max(1, len(x) // 16)

    if float(args.zoom_x_max_m) > 0:
        _m = np.isfinite(x) & (x <= float(args.zoom_x_max_m))
        xz = x[_m]
        y_phy_z = y_phy[_m]
        y_tce_z = y_tce[_m]
        y_timely_z = y_timely[_m]
    else:
        xz, y_phy_z, y_tce_z, y_timely_z = x, y_phy, y_tce, y_timely

    fig_main = rp.figures_dir / f"F8_TCE_vs_distance__{args.scenario}__ret{args.retrans}__{args.profile}__{tag}.png"

    plt.figure(figsize=(10, 6))
    plt.plot(
        x, y_phy,
        label="PHY success",
        linestyle="--",
        linewidth=2.4,
        color="#1f77b4",
        alpha=0.95,
        zorder=1
    )
    plt.plot(
        x, y_tce,
        label="TCE",
        linestyle="-",
        linewidth=2.8,
        color="#ff7f0e",
        marker="o",
        markersize=3.5,
        markevery=mark_every,
        alpha=0.98,
        zorder=3
    )
    plt.plot(
        x, y_timely,
        label="Timely success",
        linestyle="-.",
        linewidth=2.2,
        color="#2ca02c",
        marker="s",
        markersize=3.2,
        markevery=mark_every,
        alpha=0.98,
        zorder=4
    )
    plt.fill_between(
        x, y_timely, y_phy,
        color="#1f77b4",
        alpha=0.08,
        zorder=0,
        label="_nolegend_"
    )

    plt.xlabel("Distance (m)")
    plt.ylabel("Rate / Utility")
    plt.title(f"{args.scenario}: TCE vs Distance (ret={args.retrans}, {args.profile})" if not args.policy_tag else f"{args.scenario}: TCE vs Distance (ret={args.retrans}, {args.profile}, {args.policy_tag})")
    plt.ylim(0.0, y_top)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_main, dpi=240)
    plt.close()

    print(f"[OK] main plot -> {fig_main}")

    if args.save_zoom or float(args.zoom_x_max_m) > 0:
        fig_zoom = rp.figures_dir / f"F8b_TCE_vs_distance_zoom__{args.scenario}__ret{args.retrans}__{args.profile}__{tag}.png"

        plt.figure(figsize=(10, 6))
        plt.plot(
            xz, y_phy_z,
            label="PHY success",
            linestyle="--",
            linewidth=2.4,
            color="#1f77b4",
            alpha=0.95,
            zorder=1
        )
        plt.plot(
            xz, y_tce_z,
            label="TCE",
            linestyle="-",
            linewidth=2.8,
            color="#ff7f0e",
            marker="o",
            markersize=3.5,
            markevery=mark_every,
            alpha=0.98,
            zorder=3
        )
        plt.plot(
            xz, y_timely_z,
            label="Timely success",
            linestyle="-.",
            linewidth=2.2,
            color="#2ca02c",
            marker="s",
            markersize=3.2,
            markevery=mark_every,
            alpha=0.98,
            zorder=4
        )
        plt.fill_between(
            xz, y_timely_z, y_phy_z,
            color="#1f77b4",
            alpha=0.08,
            zorder=0,
            label="_nolegend_"
        )

        plt.xlabel("Distance (m)")
        plt.ylabel("Rate / Utility")
        ttl_zoom = (f"{args.scenario}: TCE vs Distance (zoom<= {int(args.zoom_x_max_m)} m, ret={args.retrans}, {args.profile})" if float(args.zoom_x_max_m) > 0 else f"{args.scenario}: TCE vs Distance (zoom, ret={args.retrans}, {args.profile})")
        if args.policy_tag:
            ttl_zoom += f", {args.policy_tag}"
        plt.title(ttl_zoom)
        plt.ylim(0.0, float(args.zoom_y_max))
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_zoom, dpi=240)
        plt.close()

        print(f"[OK] zoom plot -> {fig_zoom}")

    update_manifest(
        rp.manifest_path,
        {
            "last_tce_plot": {
                "scenario": args.scenario,
                "retrans": int(args.retrans),
                "policy_tag": str(args.policy_tag),
                "profile": args.profile,
                "source_table": src.name,
                "figure": fig_main.name,
                "save_zoom": bool(args.save_zoom),
            }
        },
    )


if __name__ == "__main__":
    main()