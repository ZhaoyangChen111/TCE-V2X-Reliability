"""Fast release checks for hardening invariants and the TCE metric."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PY_DIR = REPO / "py"
sys.path.insert(0, str(PY_DIR))

from paths_C import ensure_run_dirs_a  # noqa: E402
from run_logging import update_manifest  # noqa: E402
from modules.tce_metric import compute_tce_utility, resolve_tce_config, summarize_tce  # noqa: E402


def expect_failure(command: list[str], required_text: str) -> None:
    completed = subprocess.run(command, cwd=str(PY_DIR), text=True, capture_output=True)
    combined = completed.stdout + completed.stderr
    if completed.returncode == 0:
        raise AssertionError(f"Command unexpectedly succeeded: {' '.join(command)}")
    if required_text.lower() not in combined.lower():
        raise AssertionError(
            f"Expected failure text {required_text!r}; output was:\n{combined[-2000:]}"
        )


def check_manifest_merge() -> None:
    with tempfile.TemporaryDirectory(prefix="v2x_manifest_") as temp_dir:
        path = Path(temp_dir) / "manifest.json"
        update_manifest(path, {"pipeline": {"scenario": "Ref", "seeds": [1]}})
        update_manifest(path, {"pipeline": {"policy": "classical"}, "analysis": {"ok": True}})
        obj = json.loads(path.read_text(encoding="utf-8"))
        expected = {"scenario": "Ref", "seeds": [1], "policy": "classical"}
        if obj.get("pipeline") != expected or obj.get("analysis") != {"ok": True}:
            raise AssertionError(f"Manifest deep merge lost fields: {obj}")

    run_id = "TARGETED_MANIFEST_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    first = ensure_run_dirs_a(run_id, save_as_latest=False, meta={"script": "stage_one", "keep": 7})
    ensure_run_dirs_a(run_id, save_as_latest=False, meta={"script": "stage_two"})
    obj = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    if len(obj.get("stages", [])) != 2 or obj.get("stages", [{}])[0].get("keep") != 7:
        raise AssertionError(f"Repeated run initialization overwrote manifest history: {obj}")


def check_tce_inequality() -> None:
    packets = pd.DataFrame(
        {
            "success_phy": [1, 1, 1, 0],
            "delay_ms": [10.0, 30.0, 90.0, math.nan],
            "deadline_ms": [20.0, 20.0, 20.0, 20.0],
        }
    )
    cfg = resolve_tce_config(
        profile="custom", deadline_ms=20.0, grace_ms=50.0,
        beta=math.log(20.0), gamma=2.0, msg_rate_hz=10.0,
    )
    summary = summarize_tce(compute_tce_utility(packets, cfg))
    timely = float(summary["timely_success_rate"])
    tce = float(summary["tce"])
    phy = float(summary["phy_success_rate"])
    if not (timely <= tce <= phy):
        raise AssertionError(f"TCE inequality failed: {timely} <= {tce} <= {phy}")


def check_formal_command_transcription() -> None:
    expected = {
        "noretx": ("--rets 0", None),
        "classical": ("--rets 1,2", None),
        "nomikos": ("--rets 1,2", None),
        "udrc": ("--rets 1,2", "--udrc_lambda 0.03 --policy_cost_mode delay_cbr"),
        "mdp_lite": ("--rets 2", "--mdp_model_tag m3ref_fixed"),
    }
    for policy, required in expected.items():
        completed = subprocess.run(
            [
                sys.executable, str(REPO / "examples" / "run_paper_configuration.py"),
                "--scenario", "Ref", "--load", "NoCong", "--policy", policy,
            ],
            cwd=str(REPO), text=True, capture_output=True,
        )
        if completed.returncode != 0:
            raise AssertionError(f"Formal dry run failed for {policy}: {completed.stderr}")
        for fragment in (item for item in required if item):
            if fragment not in completed.stdout:
                raise AssertionError(f"Formal {policy} command is missing {fragment!r}")
        if "DRY RUN ONLY" not in completed.stdout:
            raise AssertionError(f"Formal helper did not remain dry-run by default for {policy}")


def check_fail_fast_paths() -> None:
    python = sys.executable
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    with tempfile.TemporaryDirectory(prefix="v2x_traj_") as temp_dir:
        trajectory = Path(temp_dir) / "traj__Ref.csv"
        pd.DataFrame(
            {
                "time_s": [0.0, 0.0, 0.1, 0.1],
                "veh_id": [1, 2, 1, 2],
                "x_m": [0.0, 10.0, 1.0, 11.0],
                "y_m": [0.0, 0.0, 0.0, 0.0],
            }
        ).to_csv(trajectory, index=False)

        common = [
            python, "sim_v2x_C.py", "--scenario", "Ref", "--retrans", "0",
            "--n_seeds", "1", "--traj_path", str(trajectory),
        ]
        expect_failure(
            common + ["--run_id", f"TARGETED_BAD_TX_{stamp}", "--tx_mode", "fixed", "--tx_id", "999"],
            "Invalid fixed tx_id",
        )
        expect_failure(
            common + [
                "--run_id", f"TARGETED_ZERO_ROWS_{stamp}", "--tx_mode", "fixed", "--tx_id", "1",
                "--collect_start_s", "999",
            ],
            "zero packet rows",
        )

    expect_failure(
        [
            python, "run_pipeline_C.py", "--run_id", f"TARGETED_MISSING_MDP_{stamp}",
            "--scenarios", "Ref", "--rets", "2", "--retx_policy", "mdp_lite",
            "--skip_gen", "--skip_plot",
        ],
        "requires --mdp_model_path",
    )


def main() -> int:
    checks = [
        check_manifest_merge,
        check_tce_inequality,
        check_formal_command_transcription,
        check_fail_fast_paths,
    ]
    for check in checks:
        check()
        print(f"[PASS] {check.__name__}")
    print("TARGETED RELEASE CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
