"""Build or execute one explicit command from the preserved S15 configuration."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", required=True, choices=["Ref", "UrbMask", "Tunnel"])
    ap.add_argument("--load", required=True, choices=["NoCong", "Cong"])
    ap.add_argument("--policy", required=True, choices=["noretx", "classical", "nomikos", "udrc", "mdp_lite"])
    ap.add_argument("--execute", action="store_true", help="Actually run the formal command; without this flag only print it.")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = json.loads((repo / "configs" / "paper_s15.json").read_text(encoding="utf-8"))
    rets = "0" if args.policy == "noretx" else ("2" if args.policy == "mdp_lite" else "1,2")
    run_id = f"S15_{args.load}_{args.scenario}_{args.policy}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd = [
        sys.executable, "run_pipeline_C.py", "--run_id", run_id,
        "--retx_policy", args.policy, "--scenarios", args.scenario, "--rets", rets,
        "--seed_start", str(cfg["seed_start"]), "--n_seeds", str(cfg["n_seeds"]),
        "--duration_s", str(cfg["duration_s"]), "--dt_s", str(cfg["dt_s"]),
        "--msg_rate_hz", str(cfg["msg_rate_hz"]), "--max_distance_m", str(cfg["max_distance_m"]),
        "--tce_profile", cfg["tce"]["profile"], "--tce_deadline_ms", str(cfg["tce"]["deadline_ms"]),
        "--tce_grace_ms", str(cfg["tce"]["grace_ms"]), "--tce_beta", str(cfg["tce"]["beta"]),
        "--tce_gamma", str(cfg["tce"]["gamma"]), "--tx_mode", cfg["tx"]["mode"],
        "--tx_k", str(cfg["tx"]["k"]), "--tx_k_cross", str(cfg["tx"]["k_cross"]),
        "--tx_cross_prefixes", cfg["tx"]["cross_prefixes"], "--skip_plot", "--save_decision_log",
    ]
    if args.load == "Cong":
        cmd.append("--enable_congestion")
    if args.policy == "udrc":
        cmd += ["--udrc_lambda", str(cfg["retransmission"]["udrc_lambda"]), "--policy_cost_mode", cfg["retransmission"]["cost_mode"]]
    if args.policy == "mdp_lite":
        model = repo / cfg["models"][f"{args.load}.{args.scenario}"]
        cmd += [
            "--mdp_model_path", str(model), "--mdp_model_tag", cfg["retransmission"]["mdp_model_tag"],
            "--mdp_threshold", str(cfg["retransmission"]["mdp_threshold"]),
            "--mdp_cost_scale", str(cfg["retransmission"]["mdp_cost_scale"]),
            "--mdp_min_samples", str(cfg["retransmission"]["mdp_min_samples"]),
            "--mdp_discount", str(cfg["retransmission"]["mdp_discount"]),
        ]

    print("Formal command (large run):")
    print(subprocess.list2cmdline(cmd))
    if not args.execute:
        print("DRY RUN ONLY. Add --execute to start the formal experiment.")
        return 0
    return subprocess.run(cmd, cwd=str(repo / "py")).returncode


if __name__ == "__main__":
    raise SystemExit(main())
