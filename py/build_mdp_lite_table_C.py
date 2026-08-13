
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paths_C import ensure_run_dirs_a, load_latest_run_id, default_run_prefix, make_run_id
from modules.mdp_lite_bins import MdpLiteBinningConfig
from modules.mdp_lite_table import (
    MdpLiteTableConfig,
    build_empirical_mdp_table,
    export_mdp_lite_state_table,
    save_mdp_lite_model,
)


def _pick_run_id(arg_run_id: str) -> str:
    s = (arg_run_id or '').strip()
    if s == '':
        rid = load_latest_run_id()
        if rid:
            return rid
        return make_run_id(prefix=default_run_prefix())
    if s.lower() == 'latest':
        rid = load_latest_run_id()
        if rid:
            return rid
    return s


def _pick_latest_file(raw_dir: Path, stem_prefix: str) -> Path:
    cands = list(raw_dir.glob(f'{stem_prefix}__seed*.csv'))
    if not cands:
        raise FileNotFoundError(f'No files match {stem_prefix}__seed*.csv in {raw_dir}')
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _filter_seeds(df: pd.DataFrame, seed_start: int, n_seeds: int) -> pd.DataFrame:
    if 'seed' not in df.columns or int(n_seeds) <= 0:
        return df.copy()
    lo = int(seed_start)
    hi = int(seed_start) + int(n_seeds) - 1
    seeds = pd.to_numeric(df['seed'], errors='coerce')
    return df[(seeds >= lo) & (seeds <= hi)].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description='Build an empirical MDP-lite transition table from reference policy runs.')
    ap.add_argument('--run_id', type=str, default='latest')
    ap.add_argument('--scenario', type=str, required=True)
    ap.add_argument('--retrans', type=int, required=True)
    ap.add_argument('--policy_tag', type=str, default='classic')
    ap.add_argument('--seed_start', type=int, default=1)
    ap.add_argument('--n_seeds', type=int, default=0, help='0 means use all seeds found in the selected files')
    ap.add_argument('--max_distance_m', type=float, default=200.0)
    ap.add_argument('--min_samples', type=int, default=3)
    ap.add_argument('--distance_edges_m', type=str, default='0,50,100,150,200')
    ap.add_argument('--slack_edges_ms', type=str, default='-1e9,-50,0,20,1e9')
    ap.add_argument('--load_edges', type=str, default='0,0.2,0.4,0.7,1.000001')
    ap.add_argument('--out_path', type=str, default='')
    args = ap.parse_args()

    rid = _pick_run_id(args.run_id)
    rp = ensure_run_dirs_a(rid, save_as_latest=False)
    raw_dir = rp.raw_dir
    pkt_path = _pick_latest_file(raw_dir, f'results_packets__{args.scenario}__ret{args.retrans}__{args.policy_tag}')
    dec_path = _pick_latest_file(raw_dir, f'results_retx_decisions__{args.scenario}__ret{args.retrans}__{args.policy_tag}')

    pkt = pd.read_csv(pkt_path)
    dec = pd.read_csv(dec_path)
    if int(args.seed_start) > 0 and int(args.n_seeds) > 0:
        lo = int(args.seed_start)
        hi = lo + int(args.n_seeds) - 1
        pkt = pkt[pd.to_numeric(pkt.get('seed', 0), errors='coerce').between(lo, hi)].copy()
        dec = dec[pd.to_numeric(dec.get('seed', 0), errors='coerce').between(lo, hi)].copy()
        seed_tag = f"seed{lo}-{hi}"
    else:
        seed_tag = "allseeds"
    if float(args.max_distance_m) > 0:
        pkt = pkt[pd.to_numeric(pkt.get('distance_m', 0), errors='coerce') <= float(args.max_distance_m)].copy()
        dec = dec[pd.to_numeric(dec.get('distance_m', 0), errors='coerce') <= float(args.max_distance_m)].copy()
    if int(args.n_seeds) > 0:
        pkt = _filter_seeds(pkt, int(args.seed_start), int(args.n_seeds))
        dec = _filter_seeds(dec, int(args.seed_start), int(args.n_seeds))
        seed_tag = f'seed{int(args.seed_start)}-{int(args.seed_start)+int(args.n_seeds)-1}'
    else:
        seed_tag = 'allseeds'

    bin_cfg = MdpLiteBinningConfig(
        distance_edges_m=tuple(float(x) for x in str(args.distance_edges_m).split(',')),
        slack_edges_ms=tuple(float(x) for x in str(args.slack_edges_ms).split(',')),
        load_edges=tuple(float(x) for x in str(args.load_edges).split(',')),
    )
    table_cfg = MdpLiteTableConfig(binning=bin_cfg, min_samples=int(args.min_samples))
    model = build_empirical_mdp_table(
        dec,
        pkt,
        scenario=str(args.scenario),
        retrans=int(args.retrans),
        source_policy_tag=str(args.policy_tag),
        table_cfg=table_cfg,
        source_run_id=str(rid),
        source_seed_tag=str(seed_tag),
    )

    out_path = Path(args.out_path) if str(args.out_path).strip() else rp.tables_dir / (
        f'mdp_lite_model__{args.scenario}__ret{args.retrans}__{args.policy_tag}__{seed_tag}.json'
    )
    save_mdp_lite_model(model, out_path)

    summary_csv = out_path.with_suffix('.csv')
    export_mdp_lite_state_table(model).to_csv(summary_csv, index=False)
    meta_txt = out_path.with_suffix('.meta.txt')
    meta_txt.write_text(
        "\n".join([
            f'run_id={rid}',
            f'scenario={args.scenario}',
            f'retrans={args.retrans}',
            f'policy_tag={args.policy_tag}',
            f'seed_tag={seed_tag}',
            f'max_distance_m={args.max_distance_m}',
            f'min_samples={args.min_samples}',
            f'n_decisions_used={model["meta"].get("n_decisions_used")}',
            f'n_exact_states={model["meta"].get("n_exact_states")}',
            f'n_coarse_rds_states={model["meta"].get("n_coarse_rds_states")}',
            f'n_coarse_rdc_states={model["meta"].get("n_coarse_rdc_states")}',
            f'n_coarse_rd_states={model["meta"].get("n_coarse_rd_states")}',
            f'n_coarse_rsc_states={model["meta"].get("n_coarse_rsc_states")}',
            f'n_coarse_rs_states={model["meta"].get("n_coarse_rs_states")}',
            f'n_coarse_rc_states={model["meta"].get("n_coarse_rc_states")}',
            f'n_coarse_r_states={model["meta"].get("n_coarse_r_states")}',
            f'global_default_samples={model["meta"].get("global_default_samples")}',
        ]),
        encoding='utf-8'
    )

    print(f'[OK] packets   : {pkt_path}')
    print(f'[OK] decisions : {dec_path}')
    print(f'[OK] model     : {out_path}')
    print(f'[OK] summary   : {summary_csv}')
    print(f'[OK] meta      : {meta_txt}')
    print(f'[INFO] decisions_used={model["meta"].get("n_decisions_used")} min_samples={model["meta"].get("min_samples")}')


if __name__ == '__main__':
    main()
