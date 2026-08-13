
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from .mdp_lite_bins import MdpLiteBinningConfig, make_state_key, state_signature, validate_binning_config


@dataclass(frozen=True)
class MdpLiteTableConfig:
    binning: MdpLiteBinningConfig = MdpLiteBinningConfig()
    min_samples: int = 3


_MODEL_CACHE: dict[str, dict] = {}
_REQUIRED_DECISION_COLS = {
    'scenario', 'retrans', 'seed', 'msg_id', 'tx_id', 'rx_id',
    'attempt_idx', 'remaining_budget', 'distance_m', 'slack_ms',
    'cbr', 'predicted_next_delay_ms', 'current_est_delay_ms',
    'decision_retransmit', 'cost_ci'
}
_REQUIRED_PACKET_COLS = {
    'scenario', 'retrans', 'seed', 'msg_id', 'tx_id', 'rx_id',
    'n_tx_attempts', 'success_phy', 'delay_ms'
}


def _safe_mean(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return float('nan')
    s = pd.to_numeric(series, errors='coerce').dropna()
    return float(s.mean()) if len(s) else float('nan')


def _sanitize_name(name: str) -> str:
    s = ''.join(ch if ch.isalnum() or ch in ('_', '-', '.') else '_' for ch in str(name))
    return s or 'model'


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{df_name} missing required columns: {missing}")


def load_mdp_lite_model(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    p = Path(str(path)).expanduser()
    if not p.exists():
        return None
    key = str(p.resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    obj = json.loads(p.read_text(encoding='utf-8'))
    _MODEL_CACHE[key] = obj
    return obj


def save_mdp_lite_model(model: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding='utf-8')
    _MODEL_CACHE[str(p.resolve())] = model
    return p


def _build_lookup_paths(state_key: str) -> list[str]:
    parts = str(state_key).split('__')
    if len(parts) != 4:
        return [state_key, 'GLOBAL_DEFAULT']
    r_part, d_part, s_part, c_part = parts
    candidates = [
        state_key,
        f'{r_part}__{d_part}__{s_part}__ANYLOAD',
        f'{r_part}__{d_part}__ANYSLACK__{c_part}',
        f'{r_part}__{d_part}__ANYSLACK__ANYLOAD',
        f'{r_part}__ANYDIST__{s_part}__{c_part}',
        f'{r_part}__ANYDIST__{s_part}__ANYLOAD',
        f'{r_part}__ANYDIST__ANYSLACK__{c_part}',
        f'{r_part}__ANYDIST__ANYSLACK__ANYLOAD',
        'GLOBAL_DEFAULT',
    ]
    seen = set()
    out = []
    for k in candidates:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out

def query_mdp_lite_transition(
    model: dict | None,
    *,
    scenario: str,
    remaining_budget: int,
    distance_m: float,
    slack_ms: float,
    cbr: float,
    min_samples: int = 3,
) -> dict | None:
    if not model:
        return None
    meta = model.get('meta', {})
    cfg = validate_binning_config(
        MdpLiteBinningConfig(
            distance_edges_m=tuple(meta.get('distance_edges_m', MdpLiteBinningConfig().distance_edges_m)),
            slack_edges_ms=tuple(meta.get('slack_edges_ms', MdpLiteBinningConfig().slack_edges_ms)),
            load_edges=tuple(meta.get('load_edges', MdpLiteBinningConfig().load_edges)),
        )
    )
    state_key = make_state_key(
        remaining_budget=int(remaining_budget),
        distance_m=float(distance_m),
        slack_ms=float(slack_ms),
        cbr=float(cbr),
        cfg=cfg,
    )
    requested_sig = state_signature(
        remaining_budget=int(remaining_budget),
        distance_m=float(distance_m),
        slack_ms=float(slack_ms),
        cbr=float(cbr),
        cfg=cfg,
    )
    states = model.get('states', {})
    search_spaces = [str(scenario), '__GLOBAL__']
    paths = _build_lookup_paths(state_key)
    effective_min_samples = max(int(min_samples), int(meta.get('min_samples', 1) or 1))
    for scope in search_spaces:
        scoped = states.get(scope, {})
        for idx, k in enumerate(paths):
            cand = scoped.get(k)
            if cand and int(cand.get('samples', 0)) >= effective_min_samples:
                out = dict(cand)
                local_level = ('exact' if idx == 0 else f'fallback_{idx}') if scope == str(scenario) else ('global_exact' if idx == 0 else f'global_fallback_{idx}')
                out['lookup_level'] = local_level
                out['lookup_scope'] = 'scenario' if scope == str(scenario) else 'global'
                out['lookup_rank'] = int(idx)
                out['state_key'] = k
                out['requested_state_key'] = state_key
                out['requested_signature'] = requested_sig
                out['effective_min_samples'] = int(effective_min_samples)
                out['hit_kind'] = 'exact' if local_level in ('exact', 'global_exact') else ('global_default' if k == 'GLOBAL_DEFAULT' else 'coarse')
                return out
    return {
        'lookup_level': 'missing',
        'lookup_scope': 'none',
        'lookup_rank': -1,
        'state_key': 'MISSING',
        'requested_state_key': state_key,
        'requested_signature': requested_sig,
        'effective_min_samples': int(effective_min_samples),
        'hit_kind': 'missing',
        'samples': 0,
    }


def _aggregate_state_rows(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(key_col, dropna=False):
        succ = pd.to_numeric(g['immediate_next_success'], errors='coerce').fillna(0)
        fail = pd.to_numeric(g['immediate_next_fail'], errors='coerce').fillna(0)
        samples = int(len(g))
        terminal_fail = float(((fail > 0.5) & g['next_current_est_delay_ms'].isna()).sum()) / max(samples, 1)
        rows.append({
            key_col: str(key),
            'samples': samples,
            'p_success_next': float(succ.mean()) if samples else float('nan'),
            'p_fail_next': float(fail.mean()) if samples else float('nan'),
            'delta_delay_succ_ms': _safe_mean(g.loc[succ > 0.5, 'delta_delay_succ_ms']),
            'delta_delay_fail_ms': _safe_mean(g.loc[fail > 0.5, 'delta_delay_fail_ms']),
            'continue_cost_ci': _safe_mean(g['continue_cost_ci']),
            'next_cbr_mean': _safe_mean(g['next_cbr_mean_obs']),
            'next_busy_pressure_mean': _safe_mean(g['next_busy_pressure_mean_obs']),
            'terminal_fail_prob': float(terminal_fail),
        })
    return pd.DataFrame(rows)


def _materialize(states_df: pd.DataFrame, key_col: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in states_df.iterrows():
        k = str(row[key_col])
        out[k] = {
            'samples': int(row.get('samples', 0)),
            'p_success_next': float(row.get('p_success_next', np.nan)),
            'p_fail_next': float(row.get('p_fail_next', np.nan)),
            'delta_delay_succ_ms': float(row.get('delta_delay_succ_ms', np.nan)),
            'delta_delay_fail_ms': float(row.get('delta_delay_fail_ms', np.nan)),
            'continue_cost_ci': float(row.get('continue_cost_ci', np.nan)),
            'next_cbr_mean': float(row.get('next_cbr_mean', np.nan)),
            'next_busy_pressure_mean': float(row.get('next_busy_pressure_mean', np.nan)),
            'terminal_fail_prob': float(row.get('terminal_fail_prob', np.nan)),
        }
    return out


def build_empirical_mdp_table(
    decision_df: pd.DataFrame,
    packet_df: pd.DataFrame,
    *,
    scenario: str,
    retrans: int,
    source_policy_tag: str,
    table_cfg: MdpLiteTableConfig | None = None,
    source_run_id: str = '',
    source_seed_tag: str = '',
) -> dict:
    table_cfg = table_cfg or MdpLiteTableConfig()
    bin_cfg = validate_binning_config(table_cfg.binning)

    dec = decision_df.copy()
    pkt = packet_df.copy()
    _ensure_columns(dec, _REQUIRED_DECISION_COLS, 'decision_df')
    _ensure_columns(pkt, _REQUIRED_PACKET_COLS, 'packet_df')

    dec = dec[(dec['scenario'] == scenario) & (pd.to_numeric(dec['retrans'], errors='coerce') == int(retrans))].copy()
    pkt = pkt[(pkt['scenario'] == scenario) & (pd.to_numeric(pkt['retrans'], errors='coerce') == int(retrans))].copy()
    if len(dec) == 0:
        raise ValueError(f'No decision rows found for scenario={scenario} retrans={retrans}')
    if len(pkt) == 0:
        raise ValueError(f'No packet rows found for scenario={scenario} retrans={retrans}')

    join_cols = ['scenario', 'retrans', 'seed', 'msg_id', 'tx_id', 'rx_id']
    pkt_sel = pkt[join_cols + ['n_tx_attempts', 'success_phy', 'delay_ms']].copy()
    merged = dec.merge(pkt_sel, on=join_cols, how='left', suffixes=('', '_pkt'))
    merged = merged[pd.to_numeric(merged.get('decision_retransmit', 0), errors='coerce').fillna(0) > 0.5].copy()

    dec_sorted = dec.sort_values(['seed', 'msg_id', 'tx_id', 'rx_id', 'attempt_idx']).copy()
    dec_sorted['attempt_idx_num'] = pd.to_numeric(dec_sorted['attempt_idx'], errors='coerce')

    # Current decision at attempt k should look up the next decision-state row at attempt k+1.
    # So the left table needs next_attempt_idx = k+1, while the right table should expose the
    # *actual* attempt index of the next decision row (not its own +1, which causes an off-by-one).
    merged['attempt_idx_num'] = pd.to_numeric(merged['attempt_idx'], errors='coerce')
    merged['next_attempt_idx'] = merged['attempt_idx_num'] + 1

    next_rows = dec_sorted[['seed', 'msg_id', 'tx_id', 'rx_id', 'attempt_idx_num', 'current_est_delay_ms', 'predicted_next_cbr', 'predicted_busy_pressure']].copy()
    next_rows = next_rows.rename(columns={
        'attempt_idx_num': 'next_attempt_idx',
        'current_est_delay_ms': 'next_current_est_delay_ms',
        'predicted_next_cbr': 'next_state_cbr',
        'predicted_busy_pressure': 'next_state_busy_pressure',
    })
    merged = merged.merge(next_rows, on=['seed', 'msg_id', 'tx_id', 'rx_id', 'next_attempt_idx'], how='left')

    merged['remaining_budget'] = pd.to_numeric(merged['remaining_budget'], errors='coerce').fillna(0).astype(int)
    cur_delay_num = pd.to_numeric(merged['current_est_delay_ms'], errors='coerce')
    if 'policy_deadline_ms' in merged.columns:
        deadline_num = pd.to_numeric(merged['policy_deadline_ms'], errors='coerce')
        merged['slack_ms_state'] = deadline_num - cur_delay_num
    elif 'deadline_ms' in merged.columns:
        deadline_num = pd.to_numeric(merged['deadline_ms'], errors='coerce')
        merged['slack_ms_state'] = deadline_num - cur_delay_num
    else:
        logged_slack = pd.to_numeric(merged['slack_ms'], errors='coerce')
        pred_next_num = pd.to_numeric(merged['predicted_next_delay_ms'], errors='coerce')
        merged['slack_ms_state'] = logged_slack + (pred_next_num - cur_delay_num)
    merged['distance_m_state'] = pd.to_numeric(merged['distance_m'], errors='coerce')
    merged['cbr_state'] = pd.to_numeric(merged['cbr'], errors='coerce').fillna(0.0)
    merged['state_key'] = merged.apply(
        lambda r: make_state_key(
            remaining_budget=int(r['remaining_budget']),
            distance_m=float(r['distance_m_state']),
            slack_ms=float(r['slack_ms_state']),
            cbr=float(r['cbr_state']),
            cfg=bin_cfg,
        ),
        axis=1,
    )

    attempt_idx = pd.to_numeric(merged['attempt_idx'], errors='coerce')
    final_attempts = pd.to_numeric(merged['n_tx_attempts'], errors='coerce')
    success_phy = pd.to_numeric(merged['success_phy'], errors='coerce').fillna(0)
    merged['immediate_next_success'] = ((success_phy > 0.5) & (final_attempts == (attempt_idx + 1))).astype(int)
    merged['immediate_next_fail'] = (1 - merged['immediate_next_success']).astype(int)

    cur_delay = pd.to_numeric(merged['current_est_delay_ms'], errors='coerce')
    pred_next_delay = pd.to_numeric(merged['predicted_next_delay_ms'], errors='coerce')
    total_delay = pd.to_numeric(merged['delay_ms'], errors='coerce')
    next_delay_state = pd.to_numeric(merged['next_current_est_delay_ms'], errors='coerce')

    succ_delta = np.where(merged['immediate_next_success'] > 0.5, total_delay - cur_delay, np.nan)
    fail_delta = np.where(next_delay_state.notna(), next_delay_state - cur_delay, pred_next_delay - cur_delay)
    merged['delta_delay_succ_ms'] = np.where(np.isfinite(succ_delta), np.maximum(succ_delta, 0.1), np.nan)
    merged['delta_delay_fail_ms'] = np.where(np.isfinite(fail_delta), np.maximum(fail_delta, 0.1), np.nan)
    merged['continue_cost_ci'] = pd.to_numeric(merged['cost_ci'], errors='coerce')
    merged['next_cbr_mean_obs'] = pd.to_numeric(merged.get('next_state_cbr', np.nan), errors='coerce')
    merged['next_busy_pressure_mean_obs'] = pd.to_numeric(merged.get('next_state_busy_pressure', np.nan), errors='coerce')

    exact = _aggregate_state_rows(merged, 'state_key')
    exact_states = _materialize(exact, 'state_key')

    parts = exact['state_key'].str.split('__', expand=True)
    coarse = exact.copy()
    coarse['r_part'] = parts[0]
    coarse['d_part'] = parts[1]
    coarse['s_part'] = parts[2]
    coarse['c_part'] = parts[3]

    def _group_to_state(df_in: pd.DataFrame, by_cols: list[str], state_builder) -> pd.DataFrame:
        grouped = df_in.groupby(by_cols, dropna=False, as_index=False).mean(numeric_only=True)
        grouped['samples'] = df_in.groupby(by_cols, dropna=False)['samples'].sum().values
        grouped['state_key'] = grouped.apply(state_builder, axis=1)
        return grouped

    coarse_rds = _group_to_state(coarse, ['r_part', 'd_part', 's_part'], lambda r: f"{r['r_part']}__{r['d_part']}__{r['s_part']}__ANYLOAD")
    coarse_rdc = _group_to_state(coarse, ['r_part', 'd_part', 'c_part'], lambda r: f"{r['r_part']}__{r['d_part']}__ANYSLACK__{r['c_part']}")
    coarse_rd = _group_to_state(coarse, ['r_part', 'd_part'], lambda r: f"{r['r_part']}__{r['d_part']}__ANYSLACK__ANYLOAD")
    coarse_rsc = _group_to_state(coarse, ['r_part', 's_part', 'c_part'], lambda r: f"{r['r_part']}__ANYDIST__{r['s_part']}__{r['c_part']}")
    coarse_rs = _group_to_state(coarse, ['r_part', 's_part'], lambda r: f"{r['r_part']}__ANYDIST__{r['s_part']}__ANYLOAD")
    coarse_rc = _group_to_state(coarse, ['r_part', 'c_part'], lambda r: f"{r['r_part']}__ANYDIST__ANYSLACK__{r['c_part']}")
    coarse_r = _group_to_state(coarse, ['r_part'], lambda r: f"{r['r_part']}__ANYDIST__ANYSLACK__ANYLOAD")

    scenario_states: Dict[str, Dict[str, Any]] = {}
    scenario_states.update(exact_states)
    scenario_states.update(_materialize(coarse_rds, 'state_key'))
    scenario_states.update(_materialize(coarse_rdc, 'state_key'))
    scenario_states.update(_materialize(coarse_rd, 'state_key'))
    scenario_states.update(_materialize(coarse_rsc, 'state_key'))
    scenario_states.update(_materialize(coarse_rs, 'state_key'))
    scenario_states.update(_materialize(coarse_rc, 'state_key'))
    scenario_states.update(_materialize(coarse_r, 'state_key'))

    global_default = {
        'samples': int(len(merged)),
        'p_success_next': float(pd.to_numeric(merged['immediate_next_success'], errors='coerce').fillna(0).mean()),
        'p_fail_next': float(pd.to_numeric(merged['immediate_next_fail'], errors='coerce').fillna(0).mean()),
        'delta_delay_succ_ms': _safe_mean(merged.loc[pd.to_numeric(merged['immediate_next_success'], errors='coerce').fillna(0) > 0.5, 'delta_delay_succ_ms']),
        'delta_delay_fail_ms': _safe_mean(merged.loc[pd.to_numeric(merged['immediate_next_fail'], errors='coerce').fillna(0) > 0.5, 'delta_delay_fail_ms']),
        'continue_cost_ci': _safe_mean(merged['continue_cost_ci']),
        'next_cbr_mean': _safe_mean(merged['next_cbr_mean_obs']),
        'next_busy_pressure_mean': _safe_mean(merged['next_busy_pressure_mean_obs']),
        'terminal_fail_prob': float(((pd.to_numeric(merged['immediate_next_fail'], errors='coerce').fillna(0) > 0.5) & merged['next_current_est_delay_ms'].isna()).sum() / max(len(merged), 1)),
    }

    model = {
        'meta': {
            'scenario': str(scenario),
            'retrans': int(retrans),
            'source_policy_tag': str(source_policy_tag),
            'source_run_id': str(source_run_id),
            'source_seed_tag': str(source_seed_tag),
            'min_samples': int(table_cfg.min_samples),
            'distance_edges_m': list(bin_cfg.distance_edges_m),
            'slack_edges_ms': list(bin_cfg.slack_edges_ms),
            'load_edges': list(bin_cfg.load_edges),
            'n_decisions_used': int(len(merged)),
            'n_exact_states': int(len(exact_states)),
            'n_coarse_rds_states': int(len(coarse_rds)),
            'n_coarse_rdc_states': int(len(coarse_rdc)),
            'n_coarse_rd_states': int(len(coarse_rd)),
            'n_coarse_rsc_states': int(len(coarse_rsc)),
            'n_coarse_rs_states': int(len(coarse_rs)),
            'n_coarse_rc_states': int(len(coarse_rc)),
            'n_coarse_r_states': int(len(coarse_r)),
            'global_default_samples': int(global_default['samples']),
        },
        'states': {
            str(scenario): scenario_states,
            '__GLOBAL__': {'GLOBAL_DEFAULT': global_default},
        },
    }
    return model


def export_mdp_lite_state_table(model: dict) -> pd.DataFrame:
    rows = []
    for scenario, states in model.get('states', {}).items():
        for k, v in states.items():
            rows.append({'scenario_key': scenario, 'state_key': k, **v})
    return pd.DataFrame(rows)
