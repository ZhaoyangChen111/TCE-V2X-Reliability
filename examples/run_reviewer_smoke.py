"""Run and validate a small end-to-end reviewer experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from datetime import datetime

import pandas as pd


REQUIRED_PACKET_COLUMNS = {
    "scenario", "retrans", "policy_tag", "retx_policy", "seed", "msg_id",
    "tx_id", "rx_id", "distance_m", "success", "success_phy", "late",
    "n_tx_attempts", "delay_ms",
}


def run_checked(command: list[str], cwd: Path) -> None:
    print("[SMOKE RUN]", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(cwd))
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with return code {completed.returncode}")


def one_file(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {pattern} in {folder}, found {len(matches)}")
    return matches[0]


def validate_policy(repo: Path, run_id: str, policy_tag: str, retrans: int, require_model: bool) -> None:
    run_dir = repo / "workspace" / "results" / "runs" / run_id
    raw_dir = run_dir / "raw"
    tables_dir = run_dir / "tables"
    packet_path = one_file(raw_dir, f"results_packets__Ref__ret{retrans}__{policy_tag}__seed1.csv")
    packets = pd.read_csv(packet_path)
    if packets.empty:
        raise AssertionError(f"Packet output is empty: {packet_path}")
    missing = REQUIRED_PACKET_COLUMNS.difference(packets.columns)
    if missing:
        raise AssertionError(f"Packet schema is missing: {sorted(missing)}")

    tce_path = one_file(tables_dir, f"tce_summary__Ref__ret{retrans}__custom__D20__G50__{policy_tag}__seed1.csv")
    summary_path = one_file(tables_dir, f"policy_summary__Ref__ret{retrans}__{policy_tag}__seed1.csv")
    tce = pd.read_csv(tce_path).iloc[0]
    policy = pd.read_csv(summary_path).iloc[0]
    timely = float(tce["timely_success_rate"])
    utility = float(tce["tce"])
    phy = float(tce["phy_success_rate"])
    if not (math.isfinite(timely) and math.isfinite(utility) and math.isfinite(phy)):
        raise AssertionError("TCE summary contains non-finite primary metrics")
    if not (timely <= utility + 1e-12 and utility <= phy + 1e-12):
        raise AssertionError(f"TCE inequality failed: Timely={timely}, TCE={utility}, PHY={phy}")

    if require_model:
        decision_path = one_file(raw_dir, f"results_retx_decisions__Ref__ret{retrans}__{policy_tag}__seed1.csv")
        decisions = pd.read_csv(decision_path)
        if decisions.empty:
            raise AssertionError("MDP smoke produced no retransmission decisions")
        hit = pd.to_numeric(decisions["mdp_model_hit"], errors="coerce").fillna(0)
        miss = pd.to_numeric(decisions["mdp_model_miss"], errors="coerce").fillna(0)
        fallback = pd.to_numeric(decisions["mdp_fallback_used"], errors="coerce").fillna(0)
        if float(hit.mean()) < 1.0 or float(miss.max()) > 0.0 or float(fallback.max()) > 0.0:
            raise AssertionError(
                f"MDP was not fully model-driven: hit={hit.mean()}, miss={miss.mean()}, fallback={fallback.mean()}"
            )
        if float(policy["mean_avg_mdp_model_hit"]) < 1.0:
            raise AssertionError("Policy summary does not confirm complete MDP model hits")

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not {"run_id", "stages", "pipeline", "effective_tce", "inputs"}.issubset(manifest):
        raise AssertionError(f"Manifest is incomplete: {manifest_path}")
    print(
        f"[SMOKE CHECK] {policy_tag}: rows={len(packets)}, "
        f"Timely={timely:.6f}, TCE={utility:.6f}, PHY={phy:.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-mdp", action="store_true", help="Also run a strict model-driven MDP-lite smoke.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    py_dir = repo / "py"
    run_id = "REVIEWER_SMOKE_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    base = [
        sys.executable, "run_pipeline_C.py", "--run_id", run_id,
        "--scenarios", "Ref", "--rets", "1", "--retx_policy", "classical",
        "--seed_start", "1", "--n_seeds", "1", "--duration_s", "2",
        "--dt_s", "0.1", "--n_vehicles", "12", "--msg_rate_hz", "10",
        "--max_distance_m", "200", "--tce_profile", "custom",
        "--tce_deadline_ms", "20", "--tce_grace_ms", "50",
        "--tce_beta", str(math.log(20.0)), "--tce_gamma", "2",
        "--tx_mode", "mix", "--tx_k", "3", "--tx_k_cross", "1",
        "--skip_plot", "--save_decision_log",
    ]
    try:
        run_checked(base, py_dir)
        validate_policy(repo, run_id, "classic", 1, require_model=False)

        if args.with_mdp:
            model = repo / "models" / "mdp_lite_model__NoCong__Ref__ret2__classic__seed1-15.json"
            mdp = [
                sys.executable, "run_pipeline_C.py", "--run_id", run_id,
                "--scenarios", "Ref", "--rets", "2", "--retx_policy", "mdp_lite",
                "--seed_start", "1", "--n_seeds", "1", "--duration_s", "2",
                "--dt_s", "0.1", "--n_vehicles", "12", "--msg_rate_hz", "10",
                "--max_distance_m", "200", "--tce_profile", "custom",
                "--tce_deadline_ms", "20", "--tce_grace_ms", "50",
                "--tce_beta", str(math.log(20.0)), "--tce_gamma", "2",
                "--tx_mode", "mix", "--tx_k", "3", "--tx_k_cross", "1",
                "--mdp_model_path", str(model), "--mdp_model_tag", "s15_nocong_ref",
                "--mdp_threshold", "0", "--mdp_cost_scale", "0.85",
                "--mdp_min_samples", "3", "--mdp_discount", "1",
                "--skip_gen", "--skip_plot", "--save_decision_log",
            ]
            run_checked(mdp, py_dir)
            validate_policy(repo, run_id, "mdplite_T0p0__C0p85__s15_nocong_ref", 2, require_model=True)

        print(f"REVIEWER SMOKE PASS: run_id={run_id}")
        return 0
    except Exception as exc:
        print(f"REVIEWER SMOKE FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
