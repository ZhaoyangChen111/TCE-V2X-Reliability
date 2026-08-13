from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd

from paths_C import ensure_run_dirs_a, make_run_id, load_latest_run_id, default_run_prefix




def _parse_tag_float(token: str, default: float = float("nan")) -> float:
    s = str(token or "").strip().replace("m", "-").replace("p", ".")
    try:
        return float(s)
    except Exception:
        return float(default)


def _policy_sort_fields(tag: str) -> tuple[int, float, float, str]:
    t = str(tag or "").strip().lower()
    if t == "noretx":
        return (0, float("nan"), float("nan"), t)
    if t == "classic":
        return (1, float("nan"), float("nan"), t)
    if t == "nomikos":
        return (2, float("nan"), float("nan"), t)
    if t.startswith("udrc_"):
        m = re.search(r"l([mp0-9]+)", t)
        lam = _parse_tag_float(m.group(1), default=float("nan")) if m else float("nan")
        return (3, lam, float("nan"), t)
    if t.startswith("mdplite_"):
        m_thr = re.search(r"_t([mp0-9]+)", t)
        m_cost = re.search(r"__c([mp0-9]+)", t)
        thr = _parse_tag_float(m_thr.group(1), default=float("nan")) if m_thr else float("nan")
        cost = _parse_tag_float(m_cost.group(1), default=float("nan")) if m_cost else float("nan")
        return (4, thr, cost, t)
    return (9, float("nan"), float("nan"), t)

def _pick_run_id(arg_run_id: str) -> str:
    s = (arg_run_id or "").strip()
    if s == "":
        return make_run_id(prefix=default_run_prefix())
    if s.lower() == "latest":
        rid = load_latest_run_id()
        return rid if rid else make_run_id(prefix=default_run_prefix())
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate policy_summary files into one comparison table")
    ap.add_argument("--run_id", type=str, default="latest")
    ap.add_argument("--scenario", required=True, choices=["Ref", "UrbMask", "Tunnel"])
    ap.add_argument("--retrans", required=True, type=int, choices=[0, 1, 2])
    args = ap.parse_args()

    run_id = _pick_run_id(args.run_id)
    rp = ensure_run_dirs_a(run_id, save_as_latest=False, meta={"script": "compare_policies_C.py"})

    cands = sorted(rp.tables_dir.glob(f"policy_summary__{args.scenario}__ret{args.retrans}__*.csv"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No policy_summary files found in {rp.tables_dir} for scenario={args.scenario} ret={args.retrans}")

    rows = []
    for p in cands:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if len(df) == 0:
            continue
        row = df.iloc[0].to_dict()
        row["source_file"] = p.name
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError("No readable rows from policy_summary files.")

    sort_fields = out["policy_tag"].map(_policy_sort_fields)
    out["_sort_family"] = sort_fields.map(lambda x: x[0])
    out["_sort_thr"] = sort_fields.map(lambda x: x[1])
    out["_sort_cost"] = sort_fields.map(lambda x: x[2])
    out["_sort_tag"] = sort_fields.map(lambda x: x[3])
    out = out.sort_values(["_sort_family", "_sort_thr", "_sort_cost", "_sort_tag", "source_file"], kind="stable").reset_index(drop=True)

    order_cols = [c for c in [
        "scenario", "retrans", "policy_tag", "retx_policy", "n_seeds",
        "mean_tce", "ci95_tce", "mean_timely_rate", "ci95_timely_rate",
        "mean_phy_rate", "ci95_phy_rate", "mean_late_ratio_phy", "ci95_late_ratio_phy",
        "mean_avg_attempts", "ci95_avg_attempts", "mean_retx_rate", "ci95_retx_rate",
        "mean_avg_gain_over_cost", "ci95_avg_gain_over_cost",
        "mean_avg_mdp_cost_scale", "ci95_avg_mdp_cost_scale",
        "mean_avg_mdp_cost_raw", "ci95_avg_mdp_cost_raw",
        "mean_avg_mdp_cost_scaled", "ci95_avg_mdp_cost_scaled",
        "mean_avg_mdp_raw_margin", "ci95_avg_mdp_raw_margin",
        "mean_avg_mdp_thresholded_margin", "ci95_avg_mdp_thresholded_margin",
        "mean_avg_mdp_model_hit", "ci95_avg_mdp_model_hit",
        "mean_avg_mdp_model_miss", "ci95_avg_mdp_model_miss",
        "mean_avg_mdp_exact_hit", "ci95_avg_mdp_exact_hit",
        "mean_avg_mdp_chain_fallback_used", "ci95_avg_mdp_chain_fallback_used",
        "mean_avg_mdp_fallback_used", "ci95_avg_mdp_fallback_used",
        "mean_share_mdp_decision_source_model", "ci95_share_mdp_decision_source_model",
        "mean_share_mdp_decision_source_chain_fallback", "ci95_share_mdp_decision_source_chain_fallback",
        "mean_share_mdp_decision_source_drop", "ci95_share_mdp_decision_source_drop",
        "mean_avg_mdp_chain_score_used", "ci95_avg_mdp_chain_score_used",
        "mean_share_mdp_abs_raw_margin_le_0p02", "ci95_share_mdp_abs_raw_margin_le_0p02",
        "mean_share_mdp_abs_thresholded_margin_le_0p02", "ci95_share_mdp_abs_thresholded_margin_le_0p02",
        "source_file"
    ] if c in out.columns]
    out = out[order_cols + [c for c in out.columns if c not in order_cols and not c.startswith("_sort_")]]

    out_path = rp.tables_dir / f"policy_compare__{args.scenario}__ret{args.retrans}.csv"
    out.to_csv(out_path, index=False)
    print(f"[OK] policy_compare -> {out_path}")


if __name__ == "__main__":
    main()
