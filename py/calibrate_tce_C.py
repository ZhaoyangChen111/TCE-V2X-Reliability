from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from paths_C import ensure_run_dirs_a, load_latest_run_id, make_run_id, default_run_prefix
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
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        p = cands[0]
        tag = p.stem.split(f'__ret{ret}__')[-1]
        return p, tag
    raise FileNotFoundError('packet file not found')


def _parse_list(s: str) -> list[float]:
    vals=[]
    for part in str(s).split(','):
        part=part.strip()
        if part:
            vals.append(float(part))
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_id', type=str, default='latest')
    ap.add_argument('--scenario', required=True, choices=['Ref','UrbMask','Tunnel'])
    ap.add_argument('--retrans', required=True, type=int, choices=[0,1,2])
    ap.add_argument('--policy_tag', type=str, default='')
    ap.add_argument('--deadlines_ms', type=str, default='20,30,40,50')
    ap.add_argument('--graces_ms', type=str, default='20,30,50')
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--gamma', type=float, default=0.0)
    ap.add_argument('--msg_rate_hz', type=float, default=10.0)
    ap.add_argument('--max_distance_m', type=float, default=200.0)
    ap.add_argument('--dist_bin_m', type=float, default=5.0)
    args = ap.parse_args()

    run_id=_pick_run_id(args.run_id)
    rp = ensure_run_dirs_a(run_id, save_as_latest=False, meta={'script':'calibrate_tce_C.py'})
    pkt_path, tag = _pick_latest_packets_file(rp.raw_dir, args.scenario, args.retrans, policy_tag=str(args.policy_tag))
    df = pd.read_csv(pkt_path)
    if float(args.max_distance_m)>0 and 'distance_m' in df.columns:
        dist=pd.to_numeric(df['distance_m'], errors='coerce')
        df=df.loc[dist.notna() & (dist<=float(args.max_distance_m))].copy()

    rows=[]
    for D in _parse_list(args.deadlines_ms):
        for G in _parse_list(args.graces_ms):
            cfg = resolve_tce_config(profile='custom', deadline_ms=D, grace_ms=G, beta=args.beta, gamma=args.gamma, msg_rate_hz=args.msg_rate_hz)
            dfu = compute_tce_utility(df, cfg=cfg, prefer_packet_deadline=False)
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
            max_gap = float((bydist['phy_success_rate'] - bydist['timely_success_rate']).abs().max()) if len(bydist) else 0.0
            sep = float((bydist['tce'] - bydist['timely_success_rate']).abs().max()) if len(bydist) else 0.0
            score = float(summ.get('late_partial_gain',0.0)) + 0.5*sep + 0.25*max_gap
            rows.append({
                'deadline_ms': D,
                'grace_ms': G,
                'timely_success_rate': float(summ.get('timely_success_rate', np.nan)),
                'phy_success_rate': float(summ.get('phy_success_rate', np.nan)),
                'tce': float(summ.get('tce', np.nan)),
                'n_late': int(summ.get('n_late', 0)),
                'late_partial_gain': float(summ.get('late_partial_gain', 0.0)),
                'max_abs_tce_minus_timely': sep,
                'max_abs_phy_minus_timely': max_gap,
                'study_score': score,
            })
    out = pd.DataFrame(rows).sort_values(['study_score','late_partial_gain','n_late'], ascending=[False,False,False])
    out_path = rp.tables_dir / f'calibrate_tce__{args.scenario}__ret{args.retrans}__{tag}.csv'
    out.to_csv(out_path, index=False)
    print(f'[OK] calibration -> {out_path}')
    print(out.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
