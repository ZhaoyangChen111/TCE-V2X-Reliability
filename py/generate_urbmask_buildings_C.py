from __future__ import annotations

import argparse

from paths_C import ensure_run_dirs_a, make_run_id, load_latest_run_id, default_run_prefix
from run_logging import log_command, update_manifest, snapshot_file

from modules.buildings_3d import generate_buildings, save_buildings_csv


def _pick_run_id(arg_run_id: str) -> str:
    s = (arg_run_id or "").strip()
    if s == "":
        return make_run_id(prefix=default_run_prefix())
    if s.lower() == "latest":
        rid = load_latest_run_id()
        return rid if rid else make_run_id(prefix=default_run_prefix())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=str, default="latest")
    ap.add_argument("--scenario", type=str, default="UrbMask")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--road_length_m", type=float, default=400.0)
    ap.add_argument("--n_blocks", type=int, default=12)
    ap.add_argument("--min_w_m", type=float, default=18.0)
    ap.add_argument("--max_w_m", type=float, default=45.0)
    ap.add_argument("--min_h_m", type=float, default=8.0)
    ap.add_argument("--max_h_m", type=float, default=22.0)
    ap.add_argument("--min_height_m", type=float, default=10.0)
    ap.add_argument("--max_height_m", type=float, default=50.0)
    ap.add_argument("--y_halfspan_mode", type=str, default="cross_road", choices=["cross_road", "one_side"])
    ap.add_argument("--x_margin_m", type=float, default=10.0)
    args = ap.parse_args()

    run_id = _pick_run_id(args.run_id)
    rp = ensure_run_dirs_a(run_id, meta={"script": "generate_urbmask_buildings_C.py", "scenario": args.scenario})
    log_command(run_id, rp.run_results_dir)

    update_manifest(
        rp.manifest_path,
        {
            "buildings": {
                "scenario": args.scenario,
                "seed": int(args.seed),
                "road_length_m": float(args.road_length_m),
                "n_blocks": int(args.n_blocks),
                "footprint": {"min_w": args.min_w_m, "max_w": args.max_w_m, "min_h": args.min_h_m, "max_h": args.max_h_m},
                "height": {"min": args.min_height_m, "max": args.max_height_m},
                "y_halfspan_mode": args.y_halfspan_mode,
                "x_margin_m": float(args.x_margin_m),
                "impl": "modules.buildings_3d.generate_buildings",
            }
        },
    )

    buildings_path = rp.buildings_dir / f"buildings__{args.scenario}__seed{args.seed}.csv"

    buildings = generate_buildings(
        seed=int(args.seed),
        road_length_m=float(args.road_length_m),
        n_blocks=int(args.n_blocks),
        x_margin_m=float(args.x_margin_m),
        y_halfspan_mode=str(args.y_halfspan_mode),
        min_w_m=float(args.min_w_m),
        max_w_m=float(args.max_w_m),
        min_h_m=float(args.min_h_m),
        max_h_m=float(args.max_h_m),
        min_height_m=float(args.min_height_m),
        max_height_m=float(args.max_height_m),
    )

    save_buildings_csv(buildings, buildings_path)
    snapshot_file(buildings_path, rp.run_results_dir, category="buildings")

    print(f"[OK] run_id={run_id}")
    print(f"[OK] buildings -> {buildings_path} (n={len(buildings)})")


if __name__ == "__main__":
    main()