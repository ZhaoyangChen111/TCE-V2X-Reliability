from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from paths_C import ensure_run_dirs_a, load_latest_run_id, make_run_id, default_run_prefix
from run_logging import log_command
from modules.tce_metric import resolve_tce_config, compute_tce_utility, summarize_tce, aggregate_tce_by_group


def _pick_run_id(arg_run_id: str) -> str:
    s = (arg_run_id or '').strip()
    if s == '':
        return make_run_id(prefix=default_run_prefix())
    if s.lower() == 'latest':
        rid = load_latest_run_id()
        return rid if rid else make_run_id(prefix=default_run_prefix())
    return s


def _pick_latest_packets_file(raw_dir: Path, scenario: str, ret: int, policy_tag: str = '') -> tuple[Path, str]:
    if (policy_tag or '').strip():
        cands = list(raw_dir.glob(f'results_packets__{scenario}__ret{ret}__{policy_tag}__seed*.csv'))
    else:
        cands = list(raw_dir.glob(f'results_packets__{scenario}__ret{ret}__seed*.csv'))
        if not cands:
            cands = list(raw_dir.glob(f'results_packets__{scenario}__ret{ret}__*.csv'))
    if cands:
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        p = cands[0]
        tag = p.stem.split(f'__ret{ret}__')[-1]
        return p, tag
    p_old = raw_dir / f'results_packets__{scenario}__ret{ret}.csv'
    if p_old.exists():
        return p_old, 'oldname'
    raise FileNotFoundError(f'Cannot find packets for scenario={scenario} ret={ret} policy_tag={policy_tag}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_id', type=str, default='latest')
    ap.add_argument('--scenario', required=True, choices=['Ref','UrbMask','Tunnel'])
    ap.add_argument('--retrans', required=True, type=int, choices=[0,1,2])
    ap.add_argument('--policy_tag', type=str, default='')
    ap.add_argument('--profile', type=str, default='safety_awareness')
    ap.add_argument('--deadline_ms', type=float, default=0.0)
    ap.add_argument('--grace_ms', type=float, default=0.0)
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--gamma', type=float, default=0.0)
    ap.add_argument('--msg_rate_hz', type=float, default=10.0)
    ap.add_argument('--prefer_packet_deadline', action='store_true')
    ap.add_argument('--max_distance_m', type=float, default=200.0)
    ap.add_argument('--dist_bin_m', type=float, default=5.0)
    args = ap.parse_args()

    run_id = _pick_run_id(args.run_id)
    rp = ensure_run_dirs_a(run_id, save_as_latest=False, meta={'script':'study_readiness_C.py'})
    log_command(run_id, rp.run_results_dir, extra=f'readiness scenario={args.scenario} ret={args.retrans}')

    pkt_path, tag = _pick_latest_packets_file(rp.raw_dir, args.scenario, args.retrans, policy_tag=str(args.policy_tag))
    df = pd.read_csv(pkt_path)
    n_raw = len(df)
    if float(args.max_distance_m) > 0 and 'distance_m' in df.columns:
        dist = pd.to_numeric(df['distance_m'], errors='coerce')
        keep = dist.notna() & (dist <= float(args.max_distance_m))
        df = df.loc[keep].copy()
        print(f'[INFO] readiness distance filter <= {float(args.max_distance_m):.1f} m: kept {len(df)}/{n_raw} packets')

    cfg = resolve_tce_config(profile=args.profile, deadline_ms=args.deadline_ms, grace_ms=args.grace_ms, beta=args.beta, gamma=args.gamma, msg_rate_hz=args.msg_rate_hz)
    dfu = compute_tce_utility(df, cfg=cfg, prefer_packet_deadline=bool(args.prefer_packet_deadline)) if len(df) else df.copy()
    summ = summarize_tce(dfu)

    if len(dfu):
        dist = pd.to_numeric(dfu.get('distance_m', np.nan), errors='coerce').to_numpy(dtype=float)
        w = max(float(args.dist_bin_m), 1e-9)
        centers = (np.floor(dist / w) + 0.5) * w
        tmp = dfu.copy()
        tmp['dist_bin_center'] = centers
        bydist = aggregate_tce_by_group(tmp, 'dist_bin_center')
    else:
        bydist = pd.DataFrame()
    max_phy_tce = float((bydist['phy_success_rate'] - bydist['tce']).abs().max()) if len(bydist) else 0.0
    max_tce_timely = float((bydist['tce'] - bydist['timely_success_rate']).abs().max()) if len(bydist) else 0.0
    max_phy_timely = float((bydist['phy_success_rate'] - bydist['timely_success_rate']).abs().max()) if len(bydist) else 0.0

    policy_tag_norm = str(tag).lower()
    is_hard_deadline_baseline = ('nomikos' in policy_tag_norm)
    ready_hidden = bool(int(summ.get('n_late', 0)) > 0 and max_tce_timely > 1e-6)
    ready_sep = bool(max_phy_timely > 1e-6)
    baseline_sane = bool(is_hard_deadline_baseline and ready_sep and int(summ.get('n_late', 0)) == 0)
    overall_phy = float(summ.get('phy_success_rate', np.nan))
    overall_tce = float(summ.get('tce', np.nan))
    overall_timely = float(summ.get('timely_success_rate', np.nan))
    if is_hard_deadline_baseline:
        status = 'BASELINE_OK' if baseline_sane else ('WEAK' if ready_sep else 'NOT_READY')
    else:
        status = 'READY' if (ready_hidden and ready_sep) else ('WEAK' if ready_sep else 'NOT_READY')

    out = pd.DataFrame([{
        'scenario': args.scenario,
        'retrans': int(args.retrans),
        'policy_tag': tag,
        'profile': cfg.profile,
        'deadline_ms': cfg.deadline_ms,
        'grace_ms': cfg.grace_ms,
        'beta': cfg.beta,
        'gamma': cfg.gamma,
        'n_packets': int(len(dfu)),
        'timely_success_rate': overall_timely,
        'phy_success_rate': overall_phy,
        'tce': overall_tce,
        'n_late': int(summ.get('n_late', 0)),
        'late_partial_gain': float(summ.get('late_partial_gain', 0.0)),
        'max_abs_phy_minus_tce': max_phy_tce,
        'max_abs_tce_minus_timely': max_tce_timely,
        'max_abs_phy_minus_timely': max_phy_timely,
        'hidden_zone_present': int(ready_hidden),
        'separation_present': int(ready_sep),
        'baseline_sane': int(baseline_sane),
        'status': status,
    }])

    profile_tag = cfg.profile
    if args.deadline_ms > 0:
        profile_tag += f'__D{int(round(cfg.deadline_ms))}'
    if args.grace_ms > 0:
        profile_tag += f'__G{int(round(cfg.grace_ms))}'
    out_path = rp.tables_dir / f'readiness__{args.scenario}__ret{args.retrans}__{profile_tag}__{tag}.csv'
    out.to_csv(out_path, index=False)
    print(f'[OK] readiness -> {out_path}')
    print(out.to_string(index=False))
    if status not in ('READY','BASELINE_OK'):
        print('[WARN] Current setup is not yet ideal for policy-study figures. Consider checking D/G or scenario pressure.')

if __name__ == '__main__':
    main()
