from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from progress_util import progress
from run_logging import update_manifest

from paths_C import (
    ensure_run_dirs_a,
    make_run_id,
    load_latest_run_id,
    ensure_base_dirs_a,
    get_base_paths_a,
    default_run_prefix,
)

from modules.mac_congestion import (
    CongestionParams,
    compute_ncs_from_distances,
    compute_airtime_s,
    compute_cbr,
    p_collision_from_ncs,
    congestion_extra_delay_ms,
)

from modules.link_variation import LinkVariationField
from modules.hotspot_weight import HotspotWeightParams, hotspot_multipliers
from modules import prop_city as pc
from modules import prop_tunnel as pt
from modules.tce_metric import resolve_tce_config
from modules.mdp_lite_table import load_mdp_lite_model
from modules.retx_policy import (
    RetxPolicyConfig,
    decide_retransmission,
    estimate_attempt_conditions,
    make_policy_tag,
)


@dataclass(frozen=True)
class Rect:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _resolve_policy_deadline_inputs(args: argparse.Namespace) -> tuple[float, float, float, float]:
    policy_deadline_ms = float(args.policy_deadline_ms)
    packet_deadline_ms = float(args.deadline_ms)
    msg_rate_hz = float(args.msg_rate_hz)
    if policy_deadline_ms <= 0.0 and packet_deadline_ms > 0.0:
        policy_deadline_ms = packet_deadline_ms
    if policy_deadline_ms <= 0.0:
        policy_deadline_ms = 50.0

    policy_grace_ms = float(args.policy_grace_ms)
    if policy_grace_ms <= 0.0:
        policy_grace_ms = min(policy_deadline_ms, 1000.0 / max(msg_rate_hz, 1e-9))

    policy_beta = float(args.policy_beta)
    if policy_beta <= 0.0:
        policy_beta = float(np.log(20.0))
    policy_gamma = float(args.policy_gamma)
    if policy_gamma <= 0.0:
        policy_gamma = 2.0
    return policy_deadline_ms, policy_grace_ms, policy_beta, policy_gamma


def load_traj(traj_path: Path) -> pd.DataFrame:
    df = pd.read_csv(traj_path)
    if "time_s" not in df.columns or "veh_id" not in df.columns:
        raise ValueError(f"trajectory missing required columns: {traj_path}")
    if "time_key" not in df.columns:
        df["time_key"] = df["time_s"].round(3)
    return df


def load_buildings(buildings_path: Path) -> list[Rect]:
    df = pd.read_csv(buildings_path)
    need = {"x_min", "x_max", "y_min", "y_max"}
    if not need.issubset(df.columns):
        raise ValueError(f"buildings missing columns {need}: {buildings_path}")
    out: list[Rect] = []
    for _, r in df.iterrows():
        out.append(Rect(float(r["x_min"]), float(r["x_max"]), float(r["y_min"]), float(r["y_max"])))
    return out


def _legacy_dirs() -> tuple[Path, Path, Path, Path]:
    ensure_base_dirs_a()
    bp = get_base_paths_a()

    traj_dir = bp.scenarios_a_dir / "trajectories"
    buildings_dir = bp.scenarios_a_dir / "buildings"
    tunnel_dir = bp.scenarios_a_dir / "tunnel"
    raw_dir = bp.results_a_root / "raw"

    # Keep trajectory/building/tunnel legacy fallbacks, but do NOT create a root-level
    # results/raw directory in 05_results_C. That folder was a harmless legacy artifact
    # which confused run inspection and wasted a small amount of space.
    for d in [traj_dir, buildings_dir, tunnel_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return traj_dir, buildings_dir, tunnel_dir, raw_dir


def _pick_run_id(arg_run_id: str) -> str:
    s = (arg_run_id or "").strip()
    if s == "":
        return make_run_id(prefix=default_run_prefix())
    if s.lower() == "latest":
        rid = load_latest_run_id()
        return rid if rid else make_run_id(prefix=default_run_prefix())
    return s


def parse_tx_ids(s: str, all_ids: Iterable[int]) -> list[int]:
    s = (s or "").strip().lower()
    ids = sorted(set(int(v) for v in all_ids))
    idset = set(ids)
    if s in ("all", "*"):
        return ids
    if s == "":
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i, b_i = int(a), int(b)
            out.extend(list(range(min(a_i, b_i), max(a_i, b_i) + 1)))
        else:
            out.append(int(part))
    out = [i for i in sorted(set(out)) if i in idset]
    return out


def _tag_is_cross(tag: str, prefixes: list[str]) -> bool:
    t = (tag or "").upper()
    return any(t.startswith(p.upper()) for p in prefixes if p)



def compute_delay_ms(
    distance_m: float,
    attempt_idx: int,
    attempt_spacing_ms: float,
    rng: np.random.Generator,
    impairment_b: float,
    extra_ms: float,
    exp_scale_ms: float,
    tail_model: str = 'exp',
) -> float:
    base = 1.0 + 0.02 * float(distance_m)
    backoff = (int(attempt_idx) - 1) * float(attempt_spacing_ms)
    jitter = float(rng.normal(0.0, 0.2))

    add = float(impairment_b) * float(extra_ms)
    tail = 0.0
    if float(impairment_b) > 1e-9 and float(exp_scale_ms) > 1e-9:
        tm = str(tail_model).lower()
        if tm == 'lognormal':
            median_ms = max(float(extra_ms), 1e-6)
            sigma = max(float(exp_scale_ms), 1e-6)
            tail = float(impairment_b) * float(rng.lognormal(mean=np.log(median_ms), sigma=sigma))
        else:
            tail = float(rng.exponential(scale=float(impairment_b) * float(exp_scale_ms)))

    return max(0.1, base + backoff + jitter + add + tail)


def simulate_one_seed(
    scenario: str,
    retrans: int,
    seed: int,
    msg_rate_hz: float,
    tx_ids_fixed: list[int],
    tx_mode: str,
    tx_k: int,
    tx_k_cross: int,
    tx_cross_prefixes: list[str],
    traj: pd.DataFrame,
    buildings: list[Rect],
    urb_transition_m: float,
    attempt_spacing_ms: float,
    fc_ghz: float,
    pl50_los_db: float,
    pl50_nlos_db: float,
    tunnel_cfg: Optional[pt.TunnelConfig],
    enable_refl_gain: bool,
    gmax_db: float,
    d0_m: float,
    refl_beta: float,
    intersection_centers_x: tuple[float, float],
    main_street_width_m: float,
    cross_street_width_m: float,
    turn_street_width_m: float,
    nlos_los_blend_m: float,
    enable_congestion: bool,
    cong: CongestionParams,
    deadline_ms: float,
    max_distance_m: float,
    collect_start_s: float,
    enable_link_variation: bool,
    link_field: Optional[LinkVariationField],
    enable_hotspot_weight: bool,
    hotspot_params: HotspotWeightParams,
    retx_cfg: RetxPolicyConfig,
    policy_tag: str,
    save_decision_log: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))

    veh_ids = np.sort(traj["veh_id"].unique()).astype(int)
    vid2i = {int(v): i for i, v in enumerate(veh_ids)}
    V = len(veh_ids)

    cols = ["time_key", "veh_id", "x_m", "y_m"]
    has_speed = "speed_mps" in traj.columns
    has_tag = "road_tag" in traj.columns
    if has_speed:
        cols.append("speed_mps")
    if has_tag:
        cols.append("road_tag")

    g = traj[cols].copy()
    time_keys = np.sort(g["time_key"].unique())

    pos_by_t: dict[float, np.ndarray] = {}
    speed_by_t: dict[float, np.ndarray] = {}
    tag_by_t: dict[float, np.ndarray] = {}

    for tk, sub in g.groupby("time_key", sort=False):
        pos = np.full((V, 2), np.nan, dtype=float)
        spd = np.full((V,), np.nan, dtype=float)
        tag = np.full((V,), "", dtype=object)

        if has_speed and has_tag:
            arr = sub[["veh_id", "x_m", "y_m", "speed_mps", "road_tag"]].to_numpy()
            for vid, x, y, s_mps, rt in arr:
                ii = vid2i[int(vid)]
                pos[ii, 0] = float(x)
                pos[ii, 1] = float(y)
                spd[ii] = float(s_mps)
                tag[ii] = str(rt)
        elif has_speed and (not has_tag):
            arr = sub[["veh_id", "x_m", "y_m", "speed_mps"]].to_numpy()
            for vid, x, y, s_mps in arr:
                ii = vid2i[int(vid)]
                pos[ii, 0] = float(x)
                pos[ii, 1] = float(y)
                spd[ii] = float(s_mps)
        elif (not has_speed) and has_tag:
            arr = sub[["veh_id", "x_m", "y_m", "road_tag"]].to_numpy()
            for vid, x, y, rt in arr:
                ii = vid2i[int(vid)]
                pos[ii, 0] = float(x)
                pos[ii, 1] = float(y)
                tag[ii] = str(rt)
        else:
            arr = sub[["veh_id", "x_m", "y_m"]].to_numpy()
            for vid, x, y in arr:
                ii = vid2i[int(vid)]
                pos[ii, 0] = float(x)
                pos[ii, 1] = float(y)

        pos_by_t[float(tk)] = pos
        if has_speed:
            speed_by_t[float(tk)] = spd
        if has_tag:
            tag_by_t[float(tk)] = tag

    t0, t1 = float(time_keys.min()), float(time_keys.max())
    dt_msg = 1.0 / float(msg_rate_hz)
    msg_times = np.arange(t0, t1 + 1e-9, dt_msg)
    msg_times = np.round(msg_times, 3)

    airtime_s = 0.0
    if enable_congestion:
        airtime_s = compute_airtime_s(
            pkt_bytes=int(cong.pkt_bytes),
            phy_rate_mbps=float(cong.phy_rate_mbps),
            mac_efficiency=float(cong.mac_efficiency),
            phy_overhead_us=float(cong.phy_overhead_us),
        )

    out_rows = []
    decision_rows = []
    msg_id = 0

    for t in progress(msg_times, total=len(msg_times), desc=f"{scenario} seed={seed}"):
        if float(t) < float(collect_start_s):
            continue

        pos = pos_by_t.get(float(t), None)
        if pos is None:
            continue

        spd = speed_by_t.get(float(t), None) if has_speed else None
        tag = tag_by_t.get(float(t), None) if has_tag else None

        active = np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1])
        active_ids = [int(veh_ids[i]) for i in range(V) if active[i]]
        if not active_ids:
            continue

        if tx_mode == "fixed":
            tx_ids = tx_ids_fixed if tx_ids_fixed else [active_ids[0]]
        elif tx_mode == "random":
            k = min(int(tx_k), len(active_ids))
            tx_ids = rng.choice(active_ids, size=k, replace=False).tolist()
        else:
            cross_ids = []
            if tag is not None:
                cross_ids = [int(veh_ids[i]) for i in range(V) if active[i] and _tag_is_cross(str(tag[i]), tx_cross_prefixes)]
            main_ids = [x for x in active_ids if x not in set(cross_ids)]

            k_cross = min(int(tx_k_cross), len(cross_ids))
            k_main = min(max(0, int(tx_k) - k_cross), len(main_ids))

            tx_ids = []
            if k_cross > 0:
                tx_ids += rng.choice(cross_ids, size=k_cross, replace=False).tolist()
            if k_main > 0:
                tx_ids += rng.choice(main_ids, size=k_main, replace=False).tolist()

            if not tx_ids:
                k = min(int(tx_k), len(active_ids))
                tx_ids = rng.choice(active_ids, size=k, replace=False).tolist()

        for tx_id in tx_ids:
            txi = vid2i.get(int(tx_id), None)
            if txi is None:
                continue

            tx_x, tx_y = float(pos[txi, 0]), float(pos[txi, 1])
            if not np.isfinite(tx_x):
                continue

            tx_speed_mps = float(spd[txi]) if (spd is not None and np.isfinite(spd[txi])) else np.nan
            tx_road_tag = str(tag[txi]) if tag is not None else ""

            dx = pos[:, 0] - tx_x
            dy = pos[:, 1] - tx_y
            dist_all = np.hypot(dx, dy)

            n_cs = 1
            cbr = 0.0
            p_col_base = 0.0
            cong_delay_ms_base = 0.0

            if enable_congestion:
                n_cs = compute_ncs_from_distances(
                    dist_all=dist_all,
                    tx_index=int(txi),
                    r_cs_m=float(cong.r_cs_m),
                    active_mask=active,
                    speed_all=spd,
                    min_speed_mps=float(cong.min_speed_mps),
                )
                cbr = compute_cbr(
                    n_cs=int(n_cs),
                    msg_rate_hz=float(msg_rate_hz),
                    airtime_s=float(airtime_s),
                    cbr_cap=float(cong.cbr_cap),
                )
                p_col_base = p_collision_from_ncs(
                    n_cs=int(n_cs),
                    alpha_col=float(cong.alpha_col),
                    cbr=float(cbr),
                    gamma_cbr_col=float(cong.gamma_cbr_col),
                    p_col_cap=float(cong.p_col_cap),
                )
                cong_delay_ms_base = congestion_extra_delay_ms(
                    rng=rng,
                    n_cs=int(n_cs),
                    beta_delay_ms=float(cong.beta_delay_ms),
                    exp_scale_ms=float(cong.exp_scale_ms),
                    cbr=float(cbr),
                    gamma_cbr_delay=float(cong.gamma_cbr_delay),
                    max_extra_delay_ms=float(cong.max_extra_delay_ms),
                )

            hotspot_mult_col = 1.0
            hotspot_mult_delay = 1.0
            if enable_hotspot_weight:
                hotspot_mult_col, hotspot_mult_delay = hotspot_multipliers(
                    road_tag=tx_road_tag,
                    tx_speed_mps=None if not np.isfinite(tx_speed_mps) else tx_speed_mps,
                    params=hotspot_params,
                )

            p_col = clamp01(p_col_base * hotspot_mult_col)
            cong_delay_ms = float(cong_delay_ms_base * hotspot_mult_delay)

            for rx_id in veh_ids:
                if int(rx_id) == int(tx_id):
                    continue
                rxi = vid2i[int(rx_id)]
                rx_x, rx_y = float(pos[rxi, 0]), float(pos[rxi, 1])
                if not np.isfinite(rx_x):
                    continue

                dist = float(dist_all[rxi])
                if (float(max_distance_m) > 0.0) and (not np.isfinite(dist) or dist > float(max_distance_m)):
                    continue

                b = 0.0
                d_min_m = float("inf")
                g_refl_db = 0.0
                tunnel_u = np.nan
                extra_ms = 0.0
                exp_scale_ms = 0.0
                tail_model = "exp"
                
                rx_road_tag = str(tag[rxi]) if tag is not None else ""

                if scenario == "Ref":
                    pl_ref = pc.urban_pathloss_los_3gpp(dist, fc_ghz=float(fc_ghz))
                    p_succ = pc.success_probability_from_outage(
                        mean_pathloss_db=pl_ref,
                        threshold_db=float(pl50_los_db),
                        sigma_sf_db=3.0,
                    )
                    link_state = "LOS"
                elif scenario == "UrbMask":
                    urb = pc.classify_urbmask_link(
                        tx_x, tx_y, rx_x, rx_y,
                        buildings=buildings,
                        transition_m=float(urb_transition_m),
                        rng=rng,
                        tx_road_tag=tx_road_tag,
                        rx_road_tag=rx_road_tag,
                        fc_ghz=float(fc_ghz),
                        pl50_los_db=float(pl50_los_db),
                        pl50_nlos_db=float(pl50_nlos_db),
                        enable_refl_gain=bool(enable_refl_gain),
                        gmax_db=float(gmax_db),
                        d0_m=float(d0_m),
                        intersection_centers_x=intersection_centers_x,
                        main_street_width_m=float(main_street_width_m),
                        cross_street_width_m=float(cross_street_width_m),
                        turn_street_width_m=float(turn_street_width_m),
                        nlos_los_blend_m=float(nlos_los_blend_m),
                    )
                    b = float(urb["blockage_b"])
                    d_min_m = float(urb["d_min_m"])
                    g_refl_db = float(urb["g_refl_db"])
                    p_succ = float(urb["p_succ"])
                    link_state = str(urb["link_state"])
                elif scenario == "Tunnel":
                    if tunnel_cfg is None:
                        raise ValueError("Tunnel scenario requires tunnel_cfg")
                    b, tunnel_u = pt.tunnel_impairment_b(tx_x, rx_x, tunnel_cfg)
                    pl_out = pc.urban_pathloss_los_3gpp(dist, fc_ghz=float(tunnel_cfg.fc_ghz))
                    pl_tun = pt.tunnel_pathloss_db(dist, tunnel_cfg)
                    pl_eff = (1.0 - float(b)) * float(pl_out) + float(b) * float(pl_tun)
                    sigma_eff = (1.0 - float(b)) * 3.0 + float(b) * float(tunnel_cfg.shadow_sigma_db)
                    p_succ = pc.success_probability_from_outage(
                        mean_pathloss_db=pl_eff,
                        threshold_db=float(pl50_los_db),
                        sigma_sf_db=float(sigma_eff),
                    )
                    extra_ms = float(tunnel_cfg.delay_extra_ms)
                    exp_scale_ms = float(tunnel_cfg.delay_exp_scale_ms)
                    tail_model = "lognormal"
                    link_state = "TUNNEL" if float(b) >= 0.15 else "LOS"
                else:
                    raise ValueError(f"Unsupported scenario: {scenario}")

                link_bias = 0.0
                if enable_link_variation and (link_field is not None):
                    link_bias = float(link_field.get_bias(int(tx_id), int(rx_id), float(t), link_state=link_state))
                    p_succ = float(np.clip(p_succ + link_bias, 0.001, 0.999))

                # Preserve the underlying simulation framework: once the scenario/link state
                # is fixed for this TX->RX pair at time t, the PHY success probability and base
                # congestion proxies are not re-written by the policy layer. Policies only decide
                # whether to continue attempting retransmissions.
                p_link_nominal = float(p_succ)
                p_attempt_actual = float(np.clip(p_link_nominal * (1.0 - float(p_col)), 0.001, 0.999)) if enable_congestion else float(np.clip(p_link_nominal, 0.001, 0.999))

                success = 0
                success_phy = 0
                late = 0
                fail_reason = "PHY_FAIL"
                n_attempts = 0
                delay_ms = np.nan
                policy_stop_reason = ""
                attempt_p_succ_eff = float(p_attempt_actual)
                attempt_cbr_eff = float(cbr)
                attempt_p_col_eff = float(p_col)
                attempt_cong_delay_eff = float(cong_delay_ms)
                attempt_busy_pressure = 0.0

                max_attempts = int(retrans) + 1
                for attempt in range(1, max_attempts + 1):
                    n_attempts = attempt
                    # Actual outcomes use the fixed scenario/channel/traffic state; the policy
                    # predictor is evaluated separately below when deciding whether to continue.
                    attempt_p_succ_eff = float(p_attempt_actual)
                    attempt_cbr_eff = float(cbr)
                    attempt_p_col_eff = float(p_col)
                    attempt_cong_delay_eff = float(cong_delay_ms)
                    attempt_busy_pressure = 0.0

                    if rng.random() < attempt_p_succ_eff:
                        success_phy = 1
                        delay_ms = compute_delay_ms(
                            dist,
                            attempt,
                            float(attempt_spacing_ms),
                            rng,
                            impairment_b=float(b),
                            extra_ms=float(extra_ms),
                            exp_scale_ms=float(exp_scale_ms),
                            tail_model=str(tail_model),
                        )
                        delay_ms = float(delay_ms + attempt_cong_delay_eff)

                        if float(deadline_ms) > 0.0 and float(delay_ms) > float(deadline_ms):
                            success = 0
                            late = 1
                            fail_reason = "DEADLINE"
                        else:
                            success = 1
                            late = 0
                            fail_reason = "OK"
                        break

                    if attempt >= max_attempts:
                        fail_reason = "PHY_FAIL"
                        break

                    dec = decide_retransmission(
                        current_attempt=int(attempt),
                        max_attempts=int(max_attempts),
                        base_link_success_prob=float(p_link_nominal),
                        distance_m=float(dist),
                        attempt_spacing_ms=float(attempt_spacing_ms),
                        impairment_b=float(b),
                        extra_ms=float(extra_ms),
                        exp_scale_ms=float(exp_scale_ms),
                        tail_model=str(tail_model),
                        base_cong_delay_ms=float(cong_delay_ms),
                        base_cbr=float(cbr),
                        base_p_col=float(p_col),
                        cfg=retx_cfg,
                        scenario=str(scenario),
                    )

                    if save_decision_log:
                        decision_rows.append(
                            [
                                scenario,
                                int(retrans),
                                str(policy_tag),
                                str(retx_cfg.policy),
                                int(seed),
                                int(msg_id),
                                float(t),
                                int(tx_id),
                                int(rx_id),
                                float(dist),
                                int(attempt),
                                int(max_attempts),
                                int(max_attempts - attempt),
                                float(dec["success_prob_next"]),
                                float(dec["current_est_delay_ms"]) if np.isfinite(dec["current_est_delay_ms"]) else np.nan,
                                float(dec["predicted_next_delay_ms"]) if np.isfinite(dec["predicted_next_delay_ms"]) else np.nan,
                                float(dec["incremental_delay_ms"]) if np.isfinite(dec["incremental_delay_ms"]) else np.nan,
                                float(dec["predicted_utility"]),
                                float(dec["expected_gain"]),
                                float(dec["cost_ci"]) if np.isfinite(dec["cost_ci"]) else np.nan,
                                float(dec["gain_over_cost"]) if np.isfinite(dec["gain_over_cost"]) else np.nan,
                                float(dec["score"]) if np.isfinite(dec["score"]) else np.nan,
                                int(dec["predicted_timely"]),
                                float(dec["slack_ms"]) if np.isfinite(dec["slack_ms"]) else np.nan,
                                str(dec.get("effective_cost_mode", retx_cfg.cost_mode)),
                                float(dec["predicted_next_cbr"]) if np.isfinite(dec["predicted_next_cbr"]) else np.nan,
                                float(dec["predicted_next_p_col"]) if np.isfinite(dec["predicted_next_p_col"]) else np.nan,
                                float(dec["predicted_next_cong_delay_ms"]) if np.isfinite(dec["predicted_next_cong_delay_ms"]) else np.nan,
                                float(dec["predicted_busy_pressure"]) if np.isfinite(dec["predicted_busy_pressure"]) else np.nan,
                                float(dec["delay_norm"]) if np.isfinite(dec["delay_norm"]) else np.nan,
                                float(dec["airtime_norm"]) if np.isfinite(dec["airtime_norm"]) else np.nan,
                                float(dec.get("delay_term", np.nan)) if np.isfinite(dec.get("delay_term", np.nan)) else np.nan,
                                float(dec.get("airtime_term", np.nan)) if np.isfinite(dec.get("airtime_term", np.nan)) else np.nan,
                                float(dec["resource_term"]) if np.isfinite(dec["resource_term"]) else np.nan,
                                float(dec["cost_multiplier"]) if np.isfinite(dec["cost_multiplier"]) else np.nan,
                                float(dec["horizon_ms"]) if np.isfinite(dec["horizon_ms"]) else np.nan,
                                float(dec["chain_expected_gain"]) if np.isfinite(dec["chain_expected_gain"]) else np.nan,
                                float(dec["chain_expected_cost"]) if np.isfinite(dec["chain_expected_cost"]) else np.nan,
                                float(dec["chain_gain_over_cost"]) if np.isfinite(dec["chain_gain_over_cost"]) else np.nan,
                                float(dec["chain_score"]) if np.isfinite(dec["chain_score"]) else np.nan,
                                float(dec["chain_best_pred_delay_ms"]) if np.isfinite(dec["chain_best_pred_delay_ms"]) else np.nan,
                                float(dec["chain_best_pred_utility"]) if np.isfinite(dec["chain_best_pred_utility"]) else np.nan,
                                int(dec["mdp_model_hit"]),
                                int(dec.get("mdp_model_miss", False)),
                                int(dec.get("mdp_exact_hit", False)),
                                int(dec.get("mdp_coarse_hit", False)),
                                int(dec.get("mdp_global_default_hit", False)),
                                str(dec.get("mdp_hit_kind", "NA")),
                                int(dec["mdp_model_samples"]),
                                int(dec.get("mdp_effective_min_samples", 0)),
                                str(dec["mdp_lookup_level"]),
                                str(dec.get("mdp_lookup_scope", "NA")),
                                int(dec.get("mdp_lookup_rank", -1)),
                                str(dec.get("mdp_state_key", "NA")),
                                str(dec.get("mdp_requested_state_key", "NA")),
                                int(dec.get("mdp_chain_fallback_used", False)),
                                int(dec.get("mdp_fallback_used", False)),
                                str(dec.get("mdp_decision_source", "NA")),
                                float(dec["mdp_q_continue"]) if np.isfinite(dec["mdp_q_continue"]) else np.nan,
                                float(dec["mdp_q_stop"]) if np.isfinite(dec["mdp_q_stop"]) else np.nan,
                                float(dec.get("mdp_cost_scale", np.nan)) if np.isfinite(dec.get("mdp_cost_scale", np.nan)) else np.nan,
                                float(dec.get("mdp_cost_raw", np.nan)) if np.isfinite(dec.get("mdp_cost_raw", np.nan)) else np.nan,
                                float(dec.get("mdp_cost_scaled", np.nan)) if np.isfinite(dec.get("mdp_cost_scaled", np.nan)) else np.nan,
                                float(dec.get("mdp_expected_success_term", np.nan)) if np.isfinite(dec.get("mdp_expected_success_term", np.nan)) else np.nan,
                                float(dec.get("mdp_future_fail_term", np.nan)) if np.isfinite(dec.get("mdp_future_fail_term", np.nan)) else np.nan,
                                float(dec.get("mdp_raw_margin", np.nan)) if np.isfinite(dec.get("mdp_raw_margin", np.nan)) else np.nan,
                                float(dec.get("mdp_threshold_applied", np.nan)) if np.isfinite(dec.get("mdp_threshold_applied", np.nan)) else np.nan,
                                float(dec.get("mdp_thresholded_margin", np.nan)) if np.isfinite(dec.get("mdp_thresholded_margin", np.nan)) else np.nan,
                                float(dec.get("mdp_chain_score_used", np.nan)) if np.isfinite(dec.get("mdp_chain_score_used", np.nan)) else np.nan,
                                float(dec["mdp_value"]) if np.isfinite(dec["mdp_value"]) else np.nan,
                                int(dec["mdp_depth_used"]),
                                int(dec["decision"]),
                                str(dec["reason"]),
                                float(deadline_ms),
                                float(max_distance_m),
                                float(retx_cfg.deadline_ms),
                                float(retx_cfg.grace_ms),
                                float(retx_cfg.lambda_cost),
                                str(retx_cfg.cost_mode),
                                int(n_cs),
                                float(cbr),
                                float(p_col),
                                float(cong_delay_ms),
                                str(link_state),
                                str(tx_road_tag),
                                str(rx_road_tag),
                            ]
                        )

                    if not bool(dec["decision"]):
                        policy_stop_reason = str(dec["reason"])
                        fail_reason = f"POLICY_DROP:{policy_stop_reason}"
                        break

                out_rows.append(
                    [
                        scenario,
                        int(retrans),
                        str(policy_tag),
                        str(retx_cfg.policy),
                        float(retx_cfg.lambda_cost),
                        str(retx_cfg.cost_mode),
                        int(seed),
                        int(msg_id),
                        float(t),
                        int(tx_id),
                        int(rx_id),
                        float(dist),
                        float(b),
                        str(link_state),
                        float(0.5 * (tx_x + rx_x)),
                        float(tunnel_u),
                        float(d_min_m),
                        float(g_refl_db),
                        int(success),
                        int(success_phy),
                        int(late),
                        str(fail_reason),
                        str(policy_stop_reason),
                        int(n_attempts),
                        float(delay_ms) if np.isfinite(delay_ms) else np.nan,
                        float(deadline_ms),
                        float(max_distance_m),
                        float(retx_cfg.deadline_ms),
                        float(retx_cfg.grace_ms),
                        int(n_cs),
                        float(cbr),
                        float(p_col),
                        float(cong_delay_ms),
                        float(attempt_p_succ_eff) if np.isfinite(attempt_p_succ_eff) else np.nan,
                        float(attempt_cbr_eff) if np.isfinite(attempt_cbr_eff) else np.nan,
                        float(attempt_p_col_eff) if np.isfinite(attempt_p_col_eff) else np.nan,
                        float(attempt_cong_delay_eff) if np.isfinite(attempt_cong_delay_eff) else np.nan,
                        float(attempt_busy_pressure) if np.isfinite(attempt_busy_pressure) else np.nan,
                        str(tx_road_tag),
                        float(link_bias),
                        float(hotspot_mult_col),
                        float(hotspot_mult_delay),
                        float(tx_speed_mps) if np.isfinite(tx_speed_mps) else np.nan,
                    ]
                )

            msg_id += 1

    pkt_df = pd.DataFrame(
        out_rows,
        columns=[
            "scenario",
            "retrans",
            "policy_tag",
            "retx_policy",
            "policy_lambda_cost",
            "policy_cost_mode",
            "seed",
            "msg_id",
            "tx_time_s",
            "tx_id",
            "rx_id",
            "distance_m",
            "blockage_b",
            "link_state",
            "mid_x_m",
            "tunnel_u",
            "d_min_m",
            "g_refl_db",
            "success",
            "success_phy",
            "late",
            "fail_reason",
            "policy_stop_reason",
            "n_tx_attempts",
            "delay_ms",
            "deadline_ms",
            "max_distance_m",
            "policy_deadline_ms",
            "policy_grace_ms",
            "n_cs",
            "cbr",
            "p_col",
            "cong_delay_ms",
            "attempt_p_success_eff",
            "attempt_cbr_eff",
            "attempt_p_col_eff",
            "attempt_cong_delay_ms_eff",
            "attempt_busy_pressure",
            "tx_road_tag",
            "link_bias",
            "hotspot_mult_col",
            "hotspot_mult_delay",
            "tx_speed_mps",
        ],
    )
    dec_df = pd.DataFrame(
        decision_rows,
        columns=[
            "scenario",
            "retrans",
            "policy_tag",
            "retx_policy",
            "seed",
            "msg_id",
            "tx_time_s",
            "tx_id",
            "rx_id",
            "distance_m",
            "attempt_idx",
            "max_attempts",
            "remaining_budget",
            "success_prob_next",
            "current_est_delay_ms",
            "predicted_next_delay_ms",
            "incremental_delay_ms",
            "predicted_utility",
            "expected_gain",
            "cost_ci",
            "gain_over_cost",
            "score",
            "predicted_timely",
            "slack_ms",
            "effective_cost_mode",
            "predicted_next_cbr",
            "predicted_next_p_col",
            "predicted_next_cong_delay_ms",
            "predicted_busy_pressure",
            "delay_norm",
            "airtime_norm",
            "delay_term",
            "airtime_term",
            "resource_term",
            "cost_multiplier",
            "cost_horizon_ms",
            "chain_expected_gain",
            "chain_expected_cost",
            "chain_gain_over_cost",
            "chain_score",
            "chain_best_pred_delay_ms",
            "chain_best_pred_utility",
            "mdp_model_hit",
            "mdp_model_miss",
            "mdp_exact_hit",
            "mdp_coarse_hit",
            "mdp_global_default_hit",
            "mdp_hit_kind",
            "mdp_model_samples",
            "mdp_effective_min_samples",
            "mdp_lookup_level",
            "mdp_lookup_scope",
            "mdp_lookup_rank",
            "mdp_state_key",
            "mdp_requested_state_key",
            "mdp_chain_fallback_used",
            "mdp_fallback_used",
            "mdp_decision_source",
            "mdp_q_continue",
            "mdp_q_stop",
            "mdp_cost_scale",
            "mdp_cost_raw",
            "mdp_cost_scaled",
            "mdp_expected_success_term",
            "mdp_future_fail_term",
            "mdp_raw_margin",
            "mdp_threshold_applied",
            "mdp_thresholded_margin",
            "mdp_chain_score_used",
            "mdp_value",
            "mdp_depth_used",
            "decision_retransmit",
            "decision_reason",
            "packet_deadline_ms",
            "max_distance_m",
            "policy_deadline_ms",
            "policy_grace_ms",
            "policy_lambda_cost",
            "policy_cost_mode",
            "n_cs",
            "cbr",
            "p_col",
            "cong_delay_ms",
            "link_state",
            "tx_road_tag",
            "rx_road_tag",
        ],
    )
    return pkt_df, dec_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, required=True, choices=["Ref", "UrbMask", "Tunnel"])
    ap.add_argument("--retrans", type=int, required=True, choices=[0, 1, 2])

    ap.add_argument("--run_id", type=str, default="")
    ap.add_argument("--seed_start", type=int, default=1)
    ap.add_argument("--n_seeds", type=int, default=1)

    ap.add_argument("--msg_rate_hz", type=float, default=10.0)

    ap.add_argument("--tx_id", type=int, default=0)
    ap.add_argument("--tx_ids", type=str, default="", help="e.g. all / 0,1,2 / 0-3")

    ap.add_argument("--tx_mode", type=str, default="fixed", choices=["fixed", "random", "mix"])
    ap.add_argument("--tx_k", type=int, default=6)
    ap.add_argument("--tx_k_cross", type=int, default=2)
    ap.add_argument("--tx_cross_prefixes", type=str, default="CROSS_,TURN_")

    ap.add_argument("--traj_path", type=str, default="")
    ap.add_argument("--traj_run_id", type=str, default="")
    ap.add_argument("--buildings_path", type=str, default="")
    ap.add_argument("--buildings_seed", type=int, default=-1)
    ap.add_argument("--transition_m", type=float, default=8.0)
    ap.add_argument("--tunnel_config_path", type=str, default="")
    ap.add_argument("--attempt_spacing_ms", type=float, default=20.0)
    ap.add_argument("--i1_x", type=float, default=1000.0)
    ap.add_argument("--i2_x", type=float, default=2000.0)
    ap.add_argument("--main_street_width_m", type=float, default=23.0)
    ap.add_argument("--cross_street_width_m", type=float, default=9.0)
    ap.add_argument("--turn_street_width_m", type=float, default=9.0)
    ap.add_argument("--nlos_los_blend_m", type=float, default=10.0)

    ap.add_argument("--fc_ghz", type=float, default=5.9)
    ap.add_argument("--pl50_los_db", type=float, default=82.5)
    ap.add_argument("--pl50_nlos_db", type=float, default=100.0)
    ap.add_argument("--pl_slope_los_db", type=float, default=2.7, help="legacy compatibility; unused in B2")
    ap.add_argument("--pl_slope_nlos_db", type=float, default=5.9, help="legacy compatibility; unused in B2")

    ap.add_argument("--enable_refl_gain", action="store_true")
    ap.add_argument("--disable_refl_gain", action="store_true")
    ap.add_argument("--gmax_db", type=float, default=6.0)
    ap.add_argument("--d0_m", type=float, default=15.0)
    ap.add_argument("--refl_beta", type=float, default=0.25)

    ap.add_argument("--enable_congestion", action="store_true")
    ap.add_argument("--cs_r_m", type=float, default=150.0)
    ap.add_argument("--cs_alpha", type=float, default=0.012)
    ap.add_argument("--cs_beta_delay_ms", type=float, default=0.10)
    ap.add_argument("--cs_exp_scale_ms", type=float, default=0.10)
    ap.add_argument("--cs_min_speed_mps", type=float, default=0.0)
    ap.add_argument("--cs_pkt_bytes", type=int, default=300)
    ap.add_argument("--cs_phy_rate_mbps", type=float, default=6.0)
    ap.add_argument("--cs_mac_efficiency", type=float, default=0.55)
    ap.add_argument("--cs_phy_overhead_us", type=float, default=300.0)
    ap.add_argument("--cs_gamma_cbr_col", type=float, default=0.40)
    ap.add_argument("--cs_gamma_cbr_delay", type=float, default=0.60)
    ap.add_argument("--cs_cbr_cap", type=float, default=0.75)

    ap.add_argument("--deadline_ms", type=float, default=0.0)
    ap.add_argument("--max_distance_m", type=float, default=200.0)
    ap.add_argument("--collect_start_s", type=float, default=0.0)

    ap.add_argument("--retx_policy", type=str, default="classical", choices=["classical", "nomikos", "udrc", "mdp_lite", "noretx"])
    ap.add_argument("--policy_deadline_ms", type=float, default=0.0)
    ap.add_argument("--policy_grace_ms", type=float, default=0.0)
    ap.add_argument("--policy_beta", type=float, default=0.0)
    ap.add_argument("--policy_gamma", type=float, default=0.0)
    ap.add_argument("--udrc_lambda", type=float, default=1.0)
    ap.add_argument("--policy_cost_mode", type=str, default="delay_cbr", choices=["delay_only", "delay_airtime", "delay_cbr", "delay_cbr_pcol"])
    ap.add_argument("--policy_airtime_weight", type=float, default=0.30)
    ap.add_argument("--policy_resource_weight", type=float, default=0.55)
    ap.add_argument("--policy_congestion_knee", type=float, default=0.35)
    ap.add_argument("--policy_attempt_load_step", type=float, default=0.08)
    ap.add_argument("--policy_attempt_col_step", type=float, default=0.12)
    ap.add_argument("--policy_attempt_delay_step", type=float, default=0.20)
    ap.add_argument("--policy_attempt_success_decay", type=float, default=0.04)
    ap.add_argument("--mdp_threshold", type=float, default=0.0)
    ap.add_argument("--mdp_cost_scale", type=float, default=1.0)
    ap.add_argument("--mdp_slack_weight", type=float, default=0.25)
    ap.add_argument("--mdp_success_weight", type=float, default=0.25)
    ap.add_argument("--mdp_model_path", type=str, default="")
    ap.add_argument("--mdp_model_tag", type=str, default="")
    ap.add_argument("--mdp_min_samples", type=int, default=3)
    ap.add_argument("--mdp_discount", type=float, default=1.0)
    ap.add_argument("--mdp_disable_chain_fallback", action="store_true")
    ap.add_argument(
        "--mdp_allow_chain_fallback",
        action="store_true",
        help="Advanced/debug only: allow chain fallback after an MDP model lookup miss.",
    )
    ap.add_argument("--save_decision_log", action="store_true")

    ap.add_argument("--enable_link_variation", action="store_true")
    ap.add_argument("--link_block_s", type=float, default=0.5)
    ap.add_argument("--link_rho", type=float, default=0.92)
    ap.add_argument("--link_sigma", type=float, default=0.04)
    ap.add_argument("--link_clip_abs", type=float, default=0.20)

    ap.add_argument("--enable_hotspot_weight", action="store_true")
    ap.add_argument("--hotspot_cross_mult", type=float, default=1.04)
    ap.add_argument("--hotspot_turn_mult", type=float, default=1.06)
    ap.add_argument("--hotspot_low_speed_thresh_mps", type=float, default=3.0)
    ap.add_argument("--hotspot_low_speed_bonus", type=float, default=0.02)
    ap.add_argument("--hotspot_queue_speed_thresh_mps", type=float, default=1.0)
    ap.add_argument("--hotspot_queue_bonus", type=float, default=0.02)

    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    def _portable_path(path: Path) -> str:
        resolved = path.expanduser().resolve()
        return resolved.relative_to(repo_root).as_posix() if resolved.is_relative_to(repo_root) else str(resolved)

    if str(args.retx_policy).lower() in ("mdp_lite", "mdplite", "mdp") and not str(args.mdp_model_path).strip():
        raise ValueError("MDP-lite requires a valid --mdp_model_path in this release.")
    if str(args.retx_policy).lower() in ("mdp_lite", "mdplite", "mdp"):
        model_candidate = Path(str(args.mdp_model_path)).expanduser().resolve()
        if not model_candidate.is_file():
            raise FileNotFoundError(f"MDP-lite model not found: {model_candidate}")

    run_id = _pick_run_id(args.run_id)
    rp = ensure_run_dirs_a(
        run_id,
        meta={"script": "sim_v2x_C.py", "scenario": args.scenario, "retrans": int(args.retrans), "retx_policy": str(args.retx_policy), "max_distance_m": float(args.max_distance_m)}
    )

    legacy_traj_dir, legacy_buildings_dir, legacy_tunnel_dir, _ = _legacy_dirs()

    if args.scenario in ("UrbMask", "Tunnel"):
        traj_name = "traj__Ref.csv"
    else:
        traj_name = f"traj__{args.scenario}.csv"

    traj_candidates = []

    if args.traj_path:
        traj_candidates.append(Path(args.traj_path))

    if args.traj_run_id:
        traj_candidates.append(
            rp.traj_dir.parent.parent / str(args.traj_run_id) / "trajectories" / traj_name
        )

    traj_candidates.append(rp.traj_dir / traj_name)
    traj_candidates.append(legacy_traj_dir / traj_name)

    traj_path = None
    for cand in traj_candidates:
        if cand.exists():
            traj_path = cand
            break

    if traj_path is None:
        searched = "\n".join(str(p) for p in traj_candidates)
        raise FileNotFoundError(
            "Trajectory file not found.\n"
            f"scenario={args.scenario}\n"
            f"expected_name={traj_name}\n"
            "Searched paths:\n"
            f"{searched}\n"
            "Please either:\n"
            "  1) run generate_trajectories_C.py with the same --run_id, or\n"
            "  2) pass --traj_path explicitly, or\n"
            "  3) pass --traj_run_id to reuse trajectories from another run."
        )

    print(f"[INFO] Using trajectory: {traj_path}")
    traj = load_traj(traj_path)
    input_info = {
        "trajectory": {
            "path": _portable_path(traj_path),
            "sha256": hashlib.sha256(traj_path.read_bytes()).hexdigest(),
        }
    }

    buildings: list[Rect] = []
    if args.scenario == "UrbMask":
        if args.buildings_path:
            bpath = Path(args.buildings_path)
        else:
            seed_use = int(args.buildings_seed) if int(args.buildings_seed) >= 0 else int(args.seed_start)
            bpath = rp.buildings_dir / f"buildings__UrbMask__seed{seed_use}.csv"
            if not bpath.exists():
                bpath = legacy_buildings_dir / f"buildings__UrbMask__seed{seed_use}.csv"
        if not bpath.exists():
            raise FileNotFoundError(
                f"UrbMask buildings file not found: {bpath}. Generate it with "
                "generate_urbmask_buildings_C.py for the same run_id or pass --buildings_path."
            )
        buildings = load_buildings(bpath)
        if not buildings:
            raise ValueError(f"UrbMask buildings file is empty: {bpath}")
        input_info["buildings"] = {
            "path": _portable_path(bpath),
            "sha256": hashlib.sha256(bpath.read_bytes()).hexdigest(),
        }

    tunnel_cfg: Optional[pt.TunnelConfig] = None
    if args.scenario == "Tunnel":
        if args.tunnel_config_path:
            tpath = Path(args.tunnel_config_path)
        else:
            tpath = rp.tunnel_dir / "tunnel_config__Tunnel.csv"
            if not tpath.exists():
                tpath = legacy_tunnel_dir / "tunnel_config__Tunnel.csv"
        if not tpath.exists():
            raise FileNotFoundError(f"tunnel config not found: {tpath}")
        tunnel_cfg = pt.TunnelConfig.from_csv(tpath)
        input_info["tunnel_config"] = {
            "path": _portable_path(tpath),
            "sha256": hashlib.sha256(tpath.read_bytes()).hexdigest(),
        }

    update_manifest(
        rp.manifest_path,
        {
            "inputs": input_info,
            "propagation_model": {
                "ref": "3gpp_urban_los_outage",
                "urbmask": {
                    "family": "geometry_plus_3gpp_virtualsource11p",
                    "same_street": "3gpp_los_nlosv_outage",
                    "cross_street": "virtualsource11p_nlos",
                    "building_blocked": "3gpp_nlos_with_optional_reflection_recovery",
                    "refs": ["3GPP TR 37.885", "Boban et al. 2014", "Mangel et al. 2011", "Abbas et al. 2013"],
                },
                "tunnel": {
                    "family": str(tunnel_cfg.model_family) if tunnel_cfg is not None else "portal_weighted_measured_logdist",
                    "refs": ["Wang et al. 2023", "Hrovat et al. 2014", "Bernado et al. 2011"],
                },
            }
        },
    )

    veh_ids = np.sort(traj["veh_id"].unique()).astype(int)
    tx_ids_fixed = parse_tx_ids(args.tx_ids, veh_ids)
    if args.tx_mode == "fixed":
        id_set = set(int(x) for x in veh_ids.tolist())
        if args.tx_ids.strip():
            if not tx_ids_fixed:
                raise ValueError(f"None of the requested fixed TX IDs exist in the trajectory: {args.tx_ids}")
        else:
            requested_tx = int(args.tx_id)
            if requested_tx not in id_set:
                raise ValueError(
                    f"Invalid fixed tx_id={requested_tx}; trajectory vehicle IDs range from "
                    f"{int(veh_ids.min())} to {int(veh_ids.max())}. Use --tx_mode mix for reviewer runs."
                )
            tx_ids_fixed = [requested_tx]

    prefixes = [p.strip() for p in str(args.tx_cross_prefixes).split(",") if p.strip()]

    cong = CongestionParams(
        r_cs_m=float(args.cs_r_m),
        alpha_col=float(args.cs_alpha),
        beta_delay_ms=float(args.cs_beta_delay_ms),
        exp_scale_ms=float(args.cs_exp_scale_ms),
        min_speed_mps=float(args.cs_min_speed_mps),
        pkt_bytes=int(args.cs_pkt_bytes),
        phy_rate_mbps=float(args.cs_phy_rate_mbps),
        mac_efficiency=float(args.cs_mac_efficiency),
        phy_overhead_us=float(args.cs_phy_overhead_us),
        gamma_cbr_col=float(args.cs_gamma_cbr_col),
        gamma_cbr_delay=float(args.cs_gamma_cbr_delay),
        cbr_cap=float(args.cs_cbr_cap),
    )

    hotspot_params = HotspotWeightParams(
        cross_mult=float(args.hotspot_cross_mult),
        turn_mult=float(args.hotspot_turn_mult),
        low_speed_thresh_mps=float(args.hotspot_low_speed_thresh_mps),
        low_speed_bonus=float(args.hotspot_low_speed_bonus),
        queue_speed_thresh_mps=float(args.hotspot_queue_speed_thresh_mps),
        queue_bonus=float(args.hotspot_queue_bonus),
    )

    enable_refl = bool(args.enable_refl_gain) and (not bool(args.disable_refl_gain))

    _policy_deadline_ms, _policy_grace_ms, _policy_beta, _policy_gamma = _resolve_policy_deadline_inputs(args)
    tce_cfg = resolve_tce_config(
        profile="custom",
        deadline_ms=float(_policy_deadline_ms),
        grace_ms=float(_policy_grace_ms),
        beta=float(_policy_beta),
        gamma=float(_policy_gamma),
        msg_rate_hz=float(args.msg_rate_hz),
    )
    airtime_ms_policy = 1000.0 * compute_airtime_s(
        pkt_bytes=int(args.cs_pkt_bytes),
        phy_rate_mbps=float(args.cs_phy_rate_mbps),
        mac_efficiency=float(args.cs_mac_efficiency),
        phy_overhead_us=float(args.cs_phy_overhead_us),
    )
    mdp_model_obj = None
    if str(args.mdp_model_path).strip():
        mdp_model_obj = load_mdp_lite_model(str(args.mdp_model_path))
        if mdp_model_obj is None:
            print(f"[WARN] mdp_model_path not found or unreadable: {args.mdp_model_path}")
        else:
            print(f"[INFO] Loaded MDP-lite model: {args.mdp_model_path}")

    retx_cfg = RetxPolicyConfig(
        policy=str(args.retx_policy),
        deadline_ms=float(tce_cfg.deadline_ms),
        grace_ms=float(tce_cfg.grace_ms),
        beta=float(tce_cfg.beta),
        gamma=float(tce_cfg.gamma),
        lambda_cost=float(args.udrc_lambda),
        cost_mode=str(args.policy_cost_mode),
        mdp_threshold=float(args.mdp_threshold),
        mdp_cost_scale=float(args.mdp_cost_scale),
        mdp_slack_weight=float(args.mdp_slack_weight),
        mdp_success_weight=float(args.mdp_success_weight),
        mdp_model_path=str(args.mdp_model_path),
        mdp_model_obj=mdp_model_obj,
        mdp_model_tag=str(args.mdp_model_tag),
        mdp_min_samples=int(args.mdp_min_samples),
        mdp_discount=float(args.mdp_discount),
        mdp_value_floor=float(max(args.mdp_threshold, 0.0)),
        mdp_fallback_to_chain=(bool(args.mdp_allow_chain_fallback) and not bool(args.mdp_disable_chain_fallback)),
        airtime_ms=float(airtime_ms_policy),
        cost_airtime_weight=float(args.policy_airtime_weight),
        cost_resource_weight=float(args.policy_resource_weight),
        congestion_knee=float(args.policy_congestion_knee),
        attempt_load_step=float(args.policy_attempt_load_step),
        attempt_col_step=float(args.policy_attempt_col_step),
        attempt_delay_step=float(args.policy_attempt_delay_step),
        attempt_success_decay=float(args.policy_attempt_success_decay),
    )
    policy_tag = make_policy_tag(
        policy=str(args.retx_policy),
        lambda_cost=float(args.udrc_lambda),
        cost_mode=str(args.policy_cost_mode),
        mdp_threshold=float(args.mdp_threshold),
        mdp_model_tag=str(args.mdp_model_tag),
        mdp_cost_scale=float(args.mdp_cost_scale),
    )

    if float(args.max_distance_m) > 0.0:
        print(f"[INFO] Applying max TX-RX distance filter: {float(args.max_distance_m):.1f} m")
    update_manifest(
        rp.manifest_path,
        {
            "simulation_scope": {
                "deadline_ms": float(args.deadline_ms),
                "max_distance_m": float(args.max_distance_m),
                "enable_congestion": bool(args.enable_congestion),
                "retx_policy": str(args.retx_policy),
                "policy_cost_mode": str(args.policy_cost_mode),
                "policy_airtime_weight": float(args.policy_airtime_weight),
                "policy_resource_weight": float(args.policy_resource_weight),
                "policy_congestion_knee": float(args.policy_congestion_knee),
                "policy_attempt_load_step": float(args.policy_attempt_load_step),
                "policy_attempt_col_step": float(args.policy_attempt_col_step),
                "policy_attempt_delay_step": float(args.policy_attempt_delay_step),
                "policy_attempt_success_decay": float(args.policy_attempt_success_decay),
                "policy_airtime_ms": float(airtime_ms_policy),
                "mdp_model_path": _portable_path(Path(args.mdp_model_path)) if str(args.mdp_model_path).strip() else "",
                "mdp_model_tag": str(args.mdp_model_tag),
                "mdp_cost_scale": float(args.mdp_cost_scale),
                "mdp_min_samples": int(args.mdp_min_samples),
                "mdp_discount": float(args.mdp_discount),
                "mdp_chain_fallback_enabled": (bool(args.mdp_allow_chain_fallback) and not bool(args.mdp_disable_chain_fallback)),
                "mdp_model_loaded": bool(mdp_model_obj is not None),
            }
        },
    )

    if (not bool(args.enable_congestion)) and str(args.retx_policy).lower() in ("udrc", "mdp_lite", "mdplite", "mdp"):
        print(
            "[INFO] enable_congestion is OFF: network-wide congestion proxies stay low, "
            "but retransmission self-load / airtime penalties still apply in v8."
        )

    seed0 = int(args.seed_start)
    n_seeds = int(args.n_seeds)
    seeds = list(range(seed0, seed0 + n_seeds))
    seed_tag = f"seed{seed0}-{seed0+n_seeds-1}" if n_seeds > 1 else f"seed{seed0}"

    frames = []
    decision_frames = []
    for sd in seeds:
        link_field = None
        if args.enable_link_variation:
            link_field = LinkVariationField(
                block_s=float(args.link_block_s),
                rho=float(args.link_rho),
                sigma=float(args.link_sigma),
                clip_abs=float(args.link_clip_abs),
                base_seed=int(sd),
            )

        pkt_df, dec_df = simulate_one_seed(
            scenario=str(args.scenario),
            retrans=int(args.retrans),
            seed=int(sd),
            msg_rate_hz=float(args.msg_rate_hz),
            tx_ids_fixed=tx_ids_fixed,
            tx_mode=str(args.tx_mode),
            tx_k=int(args.tx_k),
            tx_k_cross=int(args.tx_k_cross),
            tx_cross_prefixes=prefixes,
            traj=traj,
            buildings=buildings,
            urb_transition_m=float(args.transition_m),
            attempt_spacing_ms=float(args.attempt_spacing_ms),
            fc_ghz=float(args.fc_ghz),
            pl50_los_db=float(args.pl50_los_db),
            pl50_nlos_db=float(args.pl50_nlos_db),
            tunnel_cfg=tunnel_cfg,
            enable_refl_gain=bool(enable_refl),
            gmax_db=float(args.gmax_db),
            d0_m=float(args.d0_m),
            refl_beta=float(args.refl_beta),
            intersection_centers_x=(float(args.i1_x), float(args.i2_x)),
            main_street_width_m=float(args.main_street_width_m),
            cross_street_width_m=float(args.cross_street_width_m),
            turn_street_width_m=float(args.turn_street_width_m),
            nlos_los_blend_m=float(args.nlos_los_blend_m),
            enable_congestion=bool(args.enable_congestion),
            cong=cong,
            deadline_ms=float(args.deadline_ms),
            max_distance_m=float(args.max_distance_m),
            collect_start_s=float(args.collect_start_s),
            enable_link_variation=bool(args.enable_link_variation),
            link_field=link_field,
            enable_hotspot_weight=bool(args.enable_hotspot_weight),
            hotspot_params=hotspot_params,
            retx_cfg=retx_cfg,
            policy_tag=str(policy_tag),
            save_decision_log=bool(args.save_decision_log),
        )
        frames.append(pkt_df)
        if len(dec_df) > 0:
            decision_frames.append(dec_df)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out) == 0:
        raise RuntimeError(
            "Simulation produced zero packet rows. Check TX selection, trajectory coverage, "
            "collection window, and max_distance_m. No success marker or raw CSV was written."
        )
    out_path = rp.raw_dir / f"results_packets__{args.scenario}__ret{args.retrans}__{policy_tag}__{seed_tag}.csv"
    out.to_csv(out_path, index=False)

    dec_path = None
    if bool(args.save_decision_log):
        dec_all = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
        dec_path = rp.raw_dir / f"results_retx_decisions__{args.scenario}__ret{args.retrans}__{policy_tag}__{seed_tag}.csv"
        dec_all.to_csv(dec_path, index=False)

    update_manifest(
        rp.manifest_path,
        {
            "last_retx_policy_run": {
                "scenario": str(args.scenario),
                "retrans": int(args.retrans),
                "policy": str(args.retx_policy),
                "policy_tag": str(policy_tag),
                "policy_deadline_ms": float(retx_cfg.deadline_ms),
                "policy_grace_ms": float(retx_cfg.grace_ms),
                "policy_beta": float(retx_cfg.beta),
                "policy_gamma": float(retx_cfg.gamma),
                "udrc_lambda": float(retx_cfg.lambda_cost),
                "policy_cost_mode": str(retx_cfg.cost_mode),
                "cost_mode_runtime_note": (
                    f"{retx_cfg.cost_mode} collapses to delay_only whenever congestion proxies are zero"
                    if str(retx_cfg.cost_mode).lower() != "delay_only" else "delay_only"
                ),
                "packets_file": out_path.name,
                "decisions_file": dec_path.name if dec_path is not None else None,
            }
        },
    )

    print(f"[OK] run_id={run_id}")
    print(f"[OK] packets -> {out_path} (rows={len(out)})")
    if dec_path is not None:
        print(f"[OK] decisions -> {dec_path}")


if __name__ == "__main__":
    main()
