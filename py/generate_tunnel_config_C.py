from __future__ import annotations

import argparse
import pandas as pd

from paths_C import ensure_run_dirs_a, make_run_id, load_latest_run_id, default_run_prefix
from run_logging import log_command, update_manifest, snapshot_file

from modules.prop_tunnel import TunnelConfig


def _pick_run_id(arg_run_id: str) -> str:
    s = (arg_run_id or '').strip()
    if s == '':
        return make_run_id(prefix=default_run_prefix())
    if s.lower() == 'latest':
        rid = load_latest_run_id()
        return rid if rid else make_run_id(prefix=default_run_prefix())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_id', type=str, default='latest')
    ap.add_argument('--scenario', type=str, default='Tunnel')
    ap.add_argument('--x0_m', type=float, default=1000.0)
    ap.add_argument('--x1_m', type=float, default=2600.0)
    ap.add_argument('--transition_m', type=float, default=200.0)
    ap.add_argument('--severity', type=float, default=1.0)
    ap.add_argument('--b_floor', type=float, default=0.35)
    ap.add_argument('--b_peak', type=float, default=0.45)
    ap.add_argument('--bell_gamma', type=float, default=1.6)
    ap.add_argument('--delay_extra_ms', type=float, default=0.12)
    ap.add_argument('--delay_exp_scale_ms', type=float, default=0.25)

    ap.add_argument('--width_m', type=float, default=16.0)
    ap.add_argument('--height_m', type=float, default=7.0)
    ap.add_argument('--fc_ghz', type=float, default=5.9)
    ap.add_argument('--breakpoint_m', type=float, default=300.0)
    ap.add_argument('--near_exp', type=float, default=2.0)
    ap.add_argument('--far_exp', type=float, default=1.25)
    ap.add_argument('--shadow_sigma_db', type=float, default=2.385)
    ap.add_argument('--tunnel_extra_loss_db', type=float, default=0.0)
    ap.add_argument('--logdist_eps_db_5p2', type=float, default=50.595)
    ap.add_argument('--logdist_n_nlos', type=float, default=1.785)
    ap.add_argument('--model_family', type=str, default='portal_weighted_measured_logdist', choices=['portal_weighted_measured_logdist','segment_overlap_two_slope','two_slope'])
    args = ap.parse_args()

    if args.x1_m <= args.x0_m:
        raise ValueError('x1_m must be > x0_m')

    run_id = _pick_run_id(args.run_id)
    rp = ensure_run_dirs_a(run_id, meta={'script': 'generate_tunnel_config_C.py', 'scenario': args.scenario})
    log_command(run_id, rp.run_results_dir)

    cfg = TunnelConfig(
        x0_m=float(args.x0_m),
        x1_m=float(args.x1_m),
        transition_m=float(args.transition_m),
        severity=float(args.severity),
        b_floor=float(args.b_floor),
        b_peak=float(args.b_peak),
        bell_gamma=float(args.bell_gamma),
        width_m=float(args.width_m),
        height_m=float(args.height_m),
        fc_ghz=float(args.fc_ghz),
        breakpoint_m=float(args.breakpoint_m),
        near_exp=float(args.near_exp),
        far_exp=float(args.far_exp),
        tunnel_extra_loss_db=float(args.tunnel_extra_loss_db),
        shadow_sigma_db=float(args.shadow_sigma_db),
        logdist_eps_db_5p2=float(args.logdist_eps_db_5p2),
        logdist_n_nlos=float(args.logdist_n_nlos),
        model_family=str(args.model_family),
        delay_extra_ms=float(args.delay_extra_ms),
        delay_exp_scale_ms=float(args.delay_exp_scale_ms),
        shape=str(args.model_family),
    )

    update_manifest(
        rp.manifest_path,
        {
            'tunnel': {
                'scenario': args.scenario,
                **cfg.to_record(),
                'impl': 'modules.prop_tunnel.TunnelConfig',
            }
        },
    )

    out_path = rp.tunnel_dir / f'tunnel_config__{args.scenario}.csv'
    pd.DataFrame([cfg.to_record()]).to_csv(out_path, index=False)
    snapshot_file(out_path, rp.run_results_dir, category='tunnel')
    print(f'[OK] run_id={run_id}')
    print(f'[OK] tunnel config -> {out_path}')


if __name__ == '__main__':
    main()
