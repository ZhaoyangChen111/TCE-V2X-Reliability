from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict
from pathlib import Path

import numpy as np

from .mdp_lite_table import query_mdp_lite_transition, load_mdp_lite_model
from .mdp_lite_bins import state_signature


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _resource_pressure(cbr: float, knee: float) -> float:
    """
    Soft congestion/resource pressure proxy.

    It stays exactly zero below the knee so that low-load runs are not
    over-penalized, then rises quadratically in the high-busy region.
    """
    c = _clip01(cbr)
    k = float(np.clip(float(knee), 0.0, 0.95))
    if c <= k:
        return 0.0
    z = (c - k) / max(1e-9, 1.0 - k)
    return float(z * z)


@dataclass(frozen=True)
class RetxPolicyConfig:
    policy: str = "classical"
    deadline_ms: float = 50.0
    grace_ms: float = 50.0
    beta: float = math.log(20.0)
    gamma: float = 2.0
    lambda_cost: float = 1.0
    cost_mode: str = "delay_cbr"
    mdp_threshold: float = 0.0
    mdp_cost_scale: float = 1.0
    mdp_slack_weight: float = 0.25
    mdp_success_weight: float = 0.25
    mdp_model_path: str = ""
    mdp_model_obj: Any | None = None
    mdp_model_tag: str = ""
    mdp_min_samples: int = 3
    mdp_discount: float = 1.0
    mdp_value_floor: float = 0.0
    mdp_fallback_to_chain: bool = True

    # Added in v5: resource-aware retransmission model.
    airtime_ms: float = 0.0
    cost_airtime_weight: float = 0.30
    cost_resource_weight: float = 0.55
    congestion_knee: float = 0.35
    attempt_load_step: float = 0.08
    attempt_col_step: float = 0.12
    attempt_delay_step: float = 0.20
    attempt_success_decay: float = 0.04
    future_cost_discount: float = 0.65


def _finite_or_default(x: float, default: float = 0.0) -> float:
    try:
        xf = float(x)
    except Exception:
        return float(default)
    return float(xf) if np.isfinite(xf) else float(default)


def _nonnegative_finite(x: float, default: float = 0.0) -> float:
    return float(max(_finite_or_default(x, default=default), 0.0))


def _fmt_tag_float(x: float) -> str:
    return str(_finite_or_default(x, default=0.0)).replace("-", "m").replace(".", "p")


def make_policy_tag(
    policy: str,
    lambda_cost: float = 1.0,
    cost_mode: str = "delay_cbr",
    mdp_threshold: float = 0.0,
    mdp_model_tag: str = "",
    mdp_cost_scale: float = 1.0,
) -> str:
    pol = (policy or "classical").strip().lower()
    if pol == "classical":
        return "classic"
    if pol == "nomikos":
        return "nomikos"
    if pol == "udrc":
        lam = _fmt_tag_float(float(lambda_cost))
        return f"udrc_L{lam}__{str(cost_mode).lower()}"
    if pol in ("mdp", "mdp_lite", "mdplite"):
        thr = _fmt_tag_float(float(mdp_threshold))
        cscale = _fmt_tag_float(float(mdp_cost_scale))
        tag = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(mdp_model_tag or "").strip().lower())
        base = f"mdplite_T{thr}__C{cscale}"
        return f"{base}__{tag}" if tag else base
    if pol in ("no_retx", "noretx", "none"):
        return "noretx"
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in pol)
    return safe or "custom"


def expected_tail_ms(impairment_b: float, extra_ms: float, exp_scale_ms: float, tail_model: str) -> float:
    b = max(float(impairment_b), 0.0)
    if b <= 1e-9 or float(exp_scale_ms) <= 1e-9:
        return 0.0
    tm = str(tail_model or "exp").lower()
    if tm == "lognormal":
        median_ms = max(float(extra_ms), 1e-6)
        sigma = max(min(float(exp_scale_ms), 1.50), 1e-6)
        return float(b * math.exp(math.log(median_ms) + 0.5 * sigma * sigma))
    return float(b * max(float(exp_scale_ms), 0.0))


def estimate_attempt_delay_ms(
    distance_m: float,
    attempt_idx: int,
    attempt_spacing_ms: float,
    impairment_b: float,
    extra_ms: float,
    exp_scale_ms: float,
    tail_model: str,
    cong_delay_ms: float,
) -> float:
    base = 1.0 + 0.02 * float(distance_m)
    backoff = (max(1, int(attempt_idx)) - 1) * float(attempt_spacing_ms)
    add = float(max(impairment_b, 0.0)) * float(max(extra_ms, 0.0))
    tail = expected_tail_ms(
        impairment_b=float(impairment_b),
        extra_ms=float(extra_ms),
        exp_scale_ms=float(exp_scale_ms),
        tail_model=str(tail_model),
    )
    return max(0.1, float(base + backoff + add + tail + max(float(cong_delay_ms), 0.0)))


def utility_from_delay_ms(
    delay_ms: float,
    deadline_ms: float,
    grace_ms: float,
    beta: float,
    gamma: float,
) -> float:
    d = float(delay_ms)
    D = float(deadline_ms)
    G = max(float(grace_ms), 1e-9)
    if not np.isfinite(d):
        return 0.0
    if d <= D:
        return 1.0
    tard = d - D
    if tard > G:
        return 0.0
    u = math.exp(-float(beta) * ((tard / G) ** float(gamma)))
    return float(np.clip(u, 0.0, 1.0))


def estimate_attempt_conditions(
    *,
    attempt_idx: int,
    base_link_success_prob: float,
    base_cbr: float,
    base_p_col: float,
    base_cong_delay_ms: float,
    cfg: RetxPolicyConfig,
) -> Dict[str, float]:
    """
    Approximate how successive retransmission attempts increase system stress.

    The v4 implementation treated all future attempts as if they saw the same
    collision probability / congestion delay. That systematically favored
    classical retransmission because extra attempts added little system cost.

    v5 introduces a lightweight attempt-aware model:
    - later attempts push the effective busy ratio upward,
    - collision and queueing proxies grow with busy pressure,
    - later attempts also suffer a mild success-probability decay.
    """
    step = max(0, int(attempt_idx) - 1)
    c0 = _clip01(base_cbr)
    p0 = _clip01(base_p_col)
    d0 = max(0.0, float(base_cong_delay_ms))
    p_link = _clip01(base_link_success_prob)

    # Later attempts consume additional local airtime and aggravate busy state.
    cbr_eff = _clip01(c0 + step * float(cfg.attempt_load_step) * max(0.10, 1.0 - 0.35 * c0))
    busy = _resource_pressure(cbr_eff, cfg.congestion_knee)

    # Collision grows faster in already-busy states.
    p_col_eff = _clip01(
        p0 + step * float(cfg.attempt_col_step) * (0.25 + 0.75 * busy) * (1.0 - p0)
    )

    # Queueing/resource delay grows with attempt index and busy pressure.
    cong_delay_ms_eff = d0 * (1.0 + step * float(cfg.attempt_delay_step) * (1.0 + busy))
    if step > 0 and float(cfg.airtime_ms) > 1e-9:
        cong_delay_ms_eff += step * float(cfg.airtime_ms) * (1.0 + float(cfg.cost_resource_weight) * busy)

    # Later attempts are slightly less likely to succeed, especially when busy.
    success_scale = max(0.20, 1.0 - step * float(cfg.attempt_success_decay) * (0.75 + 0.25 * busy))
    p_success_eff = float(np.clip(p_link * (1.0 - p_col_eff) * success_scale, 0.001, 0.999))

    return {
        "cbr_eff": float(cbr_eff),
        "busy_pressure": float(busy),
        "p_col_eff": float(p_col_eff),
        "cong_delay_ms_eff": float(cong_delay_ms_eff),
        "p_success_eff": float(p_success_eff),
        "airtime_ms": float(max(0.0, float(cfg.airtime_ms))),
    }


def compute_cost_details(
    cost_mode: str,
    incremental_delay_ms: float,
    deadline_ms: float,
    grace_ms: float,
    cbr: float,
    p_col: float,
    airtime_ms: float,
    congestion_knee: float,
    cost_airtime_weight: float,
    cost_resource_weight: float,
) -> Dict[str, float | str]:
    """
    Marginal retransmission cost with explicit mode semantics.

    delay_only     : only added delay
    delay_airtime  : added delay + explicit airtime occupation
    delay_cbr      : delay/airtime scaled by busy pressure above congestion knee
    delay_cbr_pcol : delay/airtime scaled by busy pressure plus a collision proxy

    This fixes a subtle v5 issue where ``delay_only`` still silently included an
    airtime term, which made cost-mode comparisons harder to interpret.
    """
    horizon_ms = max(float(deadline_ms) + max(float(grace_ms), 0.0), float(deadline_ms), 1e-9)
    delay_norm = max(float(incremental_delay_ms), 0.0) / horizon_ms
    airtime_norm = max(float(airtime_ms), 0.0) / horizon_ms
    cbr_c = _clip01(cbr)
    pcol_c = _clip01(p_col)
    busy = _resource_pressure(cbr_c, congestion_knee)
    mode = str(cost_mode or "delay_cbr").lower()
    if mode == "delay_airtime":
        resource_term = 0.0
        eff_mode = "delay_airtime"
        delay_term = delay_norm
        airtime_term = float(cost_airtime_weight) * airtime_norm
    elif mode == "delay_only":
        resource_term = 0.0
        eff_mode = "delay_only"
        delay_term = delay_norm
        airtime_term = 0.0
    elif mode == "delay_cbr_pcol":
        resource_term = float(busy + 0.75 * pcol_c)
        eff_mode = "delay_cbr_pcol" if resource_term > 1e-12 else "delay_airtime"
        delay_term = delay_norm * (1.0 + resource_term)
        airtime_term = float(cost_airtime_weight) * airtime_norm * (1.0 + float(cost_resource_weight) * resource_term)
    else:
        resource_term = float(busy)
        eff_mode = "delay_cbr" if resource_term > 1e-12 else "delay_airtime"
        delay_term = delay_norm * (1.0 + resource_term)
        airtime_term = float(cost_airtime_weight) * airtime_norm * (1.0 + float(cost_resource_weight) * resource_term)

    cost_ci = float(delay_term + airtime_term)
    return {
        "cost_ci": float(cost_ci),
        "delay_norm": float(delay_norm),
        "airtime_norm": float(airtime_norm),
        "resource_term": float(resource_term),
        "busy_pressure": float(busy),
        "cost_multiplier": float(1.0 + resource_term),
        "effective_cost_mode": str(eff_mode),
        "horizon_ms": float(horizon_ms),
        "delay_term": float(delay_term),
        "airtime_term": float(airtime_term),
    }


def _expected_chain_metrics(
    *,
    current_attempt: int,
    max_attempts: int,
    base_link_success_prob: float,
    distance_m: float,
    attempt_spacing_ms: float,
    impairment_b: float,
    extra_ms: float,
    exp_scale_ms: float,
    tail_model: str,
    base_cong_delay_ms: float,
    base_cbr: float,
    base_p_col: float,
    cfg: RetxPolicyConfig,
    pred_cur_delay: float,
) -> Dict[str, float | str]:
    if current_attempt >= max_attempts:
        return {
            "chain_expected_gain": 0.0,
            "chain_expected_cost": 0.0,
            "chain_gain_over_cost": np.nan,
            "chain_score": np.nan,
            "chain_best_pred_delay_ms": np.nan,
            "chain_best_pred_utility": 0.0,
            "chain_reach_last_attempt_prob": 0.0,
            "effective_cost_mode": str((cfg.cost_mode or "delay_only").lower()),
            "delay_norm": np.nan,
            "airtime_norm": np.nan,
            "delay_term": np.nan,
            "airtime_term": np.nan,
            "resource_term": np.nan,
            "busy_pressure": np.nan,
            "cost_multiplier": np.nan,
            "horizon_ms": np.nan,
        }

    total_gain = 0.0
    total_cost = 0.0
    reach_prob = 1.0
    best_u = -1.0
    best_delay = np.nan
    prev_delay = float(pred_cur_delay)
    first_cost_eff_mode = str((cfg.cost_mode or "delay_only").lower())
    first_delay_norm = np.nan
    first_airtime_norm = np.nan
    first_resource_term = np.nan
    first_busy_pressure = np.nan
    first_cost_multiplier = np.nan
    first_horizon = np.nan
    reach_last_prob = 0.0

    for a in range(int(current_attempt) + 1, int(max_attempts) + 1):
        cond = estimate_attempt_conditions(
            attempt_idx=int(a),
            base_link_success_prob=float(base_link_success_prob),
            base_cbr=float(base_cbr),
            base_p_col=float(base_p_col),
            base_cong_delay_ms=float(base_cong_delay_ms),
            cfg=cfg,
        )
        pred_delay = estimate_attempt_delay_ms(
            distance_m=float(distance_m),
            attempt_idx=int(a),
            attempt_spacing_ms=float(attempt_spacing_ms),
            impairment_b=float(impairment_b),
            extra_ms=float(extra_ms),
            exp_scale_ms=float(exp_scale_ms),
            tail_model=str(tail_model),
            cong_delay_ms=float(cond["cong_delay_ms_eff"]),
        )
        inc_delay = max(0.0, float(pred_delay - prev_delay))
        util = utility_from_delay_ms(
            delay_ms=float(pred_delay),
            deadline_ms=float(cfg.deadline_ms),
            grace_ms=float(cfg.grace_ms),
            beta=float(cfg.beta),
            gamma=float(cfg.gamma),
        )
        cost_info = compute_cost_details(
            cost_mode=str(cfg.cost_mode),
            incremental_delay_ms=float(inc_delay),
            deadline_ms=float(cfg.deadline_ms),
            grace_ms=float(cfg.grace_ms),
            cbr=float(cond["cbr_eff"]),
            p_col=float(cond["p_col_eff"]),
            airtime_ms=float(cond["airtime_ms"]),
            congestion_knee=float(cfg.congestion_knee),
            cost_airtime_weight=float(cfg.cost_airtime_weight),
            cost_resource_weight=float(cfg.cost_resource_weight),
        )

        p_a = float(cond["p_success_eff"])
        success_at_a = reach_prob * p_a
        step_idx = a - (int(current_attempt) + 1)
        future_w = float(cfg.future_cost_discount) ** max(step_idx, 0)
        total_gain += success_at_a * float(util)
        total_cost += reach_prob * float(cost_info["cost_ci"]) * future_w

        if a == int(current_attempt) + 1:
            first_cost_eff_mode = str(cost_info["effective_cost_mode"])
            first_delay_norm = float(cost_info["delay_norm"])
            first_airtime_norm = float(cost_info["airtime_norm"])
            first_resource_term = float(cost_info["resource_term"])
            first_busy_pressure = float(cost_info["busy_pressure"])
            first_cost_multiplier = float(cost_info["cost_multiplier"])
            first_horizon = float(cost_info["horizon_ms"])

        if float(util) > best_u:
            best_u = float(util)
            best_delay = float(pred_delay)

        reach_last_prob = reach_prob
        reach_prob *= (1.0 - p_a)
        prev_delay = float(pred_delay)

    chain_score = float(total_gain - float(cfg.lambda_cost) * total_cost)
    chain_goverc = float(total_gain / max(total_cost, 1e-9)) if total_cost > 0 else np.nan
    return {
        "chain_expected_gain": float(total_gain),
        "chain_expected_cost": float(total_cost),
        "chain_gain_over_cost": float(chain_goverc),
        "chain_score": float(chain_score),
        "chain_best_pred_delay_ms": float(best_delay) if np.isfinite(best_delay) else np.nan,
        "chain_best_pred_utility": float(max(best_u, 0.0)),
        "chain_reach_last_attempt_prob": float(reach_last_prob),
        "effective_cost_mode": str(first_cost_eff_mode),
        "delay_norm": float(first_delay_norm) if np.isfinite(first_delay_norm) else np.nan,
        "airtime_norm": float(first_airtime_norm) if np.isfinite(first_airtime_norm) else np.nan,
        "resource_term": float(first_resource_term) if np.isfinite(first_resource_term) else np.nan,
        "busy_pressure": float(first_busy_pressure) if np.isfinite(first_busy_pressure) else np.nan,
        "cost_multiplier": float(first_cost_multiplier) if np.isfinite(first_cost_multiplier) else np.nan,
        "horizon_ms": float(first_horizon) if np.isfinite(first_horizon) else np.nan,
    }



def _default_mdp_model_tag(model_path: str, explicit_tag: str = "") -> str:
    tag = str(explicit_tag or "").strip()
    if tag:
        return tag
    mp = str(model_path or "").strip()
    if not mp:
        return ""
    stem = Path(mp).stem.lower()
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stem)
    return safe[:24]


def _mdp_lite_value(
    *,
    scenario: str,
    remaining_budget: int,
    distance_m: float,
    current_delay_ms: float,
    cbr: float,
    cfg: RetxPolicyConfig,
    model: dict | None,
    cache: dict[tuple, dict],
    depth: int = 0,
) -> dict[str, float | str | int | bool]:
    key = (str(scenario), int(remaining_budget), round(float(distance_m), 3), round(float(current_delay_ms), 3), round(float(cbr), 4), int(depth))
    if key in cache:
        return cache[key]

    if int(remaining_budget) <= 0 or float(current_delay_ms) > float(cfg.deadline_ms + cfg.grace_ms):
        out = {
            "value": 0.0,
            "hit": False,
            "samples": 0,
            "effective_min_samples": int(cfg.mdp_min_samples),
            "lookup_level": "terminal",
            "p_success_next": 0.0,
            "continue_cost_ci": 0.0,
            "delta_delay_succ_ms": 0.0,
            "delta_delay_fail_ms": 0.0,
            "depth_used": int(depth),
            "state_key": "TERMINAL",
            "requested_state_key": "TERMINAL",
            "requested_signature": {},
            "lookup_scope": "terminal",
            "lookup_rank": -1,
            "hit_kind": "terminal",
        }
        cache[key] = out
        return out

    slack_ms = float(cfg.deadline_ms - current_delay_ms)
    trans = query_mdp_lite_transition(
        model,
        scenario=str(scenario),
        remaining_budget=int(remaining_budget),
        distance_m=float(distance_m),
        slack_ms=float(slack_ms),
        cbr=float(cbr),
        min_samples=int(cfg.mdp_min_samples),
    )
    if not trans or str(trans.get("hit_kind", "missing")) == "missing":
        req_key = str((trans or {}).get("requested_state_key", "MISSING"))
        req_sig = dict((trans or {}).get("requested_signature", {}))
        out = {
            "value": float("nan"),
            "hit": False,
            "samples": 0,
            "effective_min_samples": int((trans or {}).get("effective_min_samples", cfg.mdp_min_samples)),
            "lookup_level": str((trans or {}).get("lookup_level", "missing")),
            "lookup_scope": str((trans or {}).get("lookup_scope", "none")),
            "lookup_rank": int((trans or {}).get("lookup_rank", -1)),
            "hit_kind": str((trans or {}).get("hit_kind", "missing")),
            "p_success_next": float("nan"),
            "continue_cost_ci": float("nan"),
            "delta_delay_succ_ms": float("nan"),
            "delta_delay_fail_ms": float("nan"),
            "depth_used": int(depth),
            "state_key": req_key,
            "requested_state_key": req_key,
            "requested_signature": req_sig,
        }
        cache[key] = out
        return out

    p_succ = _clip01(float(trans.get("p_success_next", np.nan)))
    d_succ = max(float(trans.get("delta_delay_succ_ms", np.nan)), 0.1)
    d_fail = max(float(trans.get("delta_delay_fail_ms", np.nan)), 0.1)
    cost_ci = max(float(trans.get("continue_cost_ci", np.nan)), 0.0)
    cost_scale = _nonnegative_finite(cfg.mdp_cost_scale, default=1.0)
    scaled_cost_ci = float(cost_scale * cost_ci)
    next_cbr = float(trans.get("next_cbr_mean", cbr)) if np.isfinite(float(trans.get("next_cbr_mean", cbr))) else float(cbr)

    util_succ = utility_from_delay_ms(
        delay_ms=float(current_delay_ms + d_succ),
        deadline_ms=float(cfg.deadline_ms),
        grace_ms=float(cfg.grace_ms),
        beta=float(cfg.beta),
        gamma=float(cfg.gamma),
    )
    child = _mdp_lite_value(
        scenario=str(scenario),
        remaining_budget=int(remaining_budget) - 1,
        distance_m=float(distance_m),
        current_delay_ms=float(current_delay_ms + d_fail),
        cbr=float(next_cbr),
        cfg=cfg,
        model=model,
        cache=cache,
        depth=int(depth) + 1,
    )
    success_term = float(p_succ * float(util_succ))
    future_fail_term = float((1.0 - p_succ) * float(cfg.mdp_discount) * float(child["value"]))
    q_continue = -float(cfg.lambda_cost) * scaled_cost_ci + success_term + future_fail_term
    out = {
        "value": float(max(0.0, q_continue)),
        "q_continue": float(q_continue),
        "q_stop": 0.0,
        "hit": True,
        "samples": int(trans.get("samples", 0)),
        "lookup_level": str(trans.get("lookup_level", "exact")),
        "lookup_scope": str(trans.get("lookup_scope", "scenario")),
        "lookup_rank": int(trans.get("lookup_rank", 0)),
        "requested_state_key": str(trans.get("requested_state_key", "NA")),
        "requested_signature": dict(trans.get("requested_signature", {})),
        "effective_min_samples": int(trans.get("effective_min_samples", cfg.mdp_min_samples)),
        "hit_kind": str(trans.get("hit_kind", "exact")),
        "p_success_next": float(p_succ),
        "continue_cost_ci": float(cost_ci),
        "continue_cost_ci_scaled": float(scaled_cost_ci),
        "cost_scale": float(cost_scale),
        "success_term": float(success_term),
        "future_fail_term": float(future_fail_term),
        "delta_delay_succ_ms": float(d_succ),
        "delta_delay_fail_ms": float(d_fail),
        "depth_used": max(int(depth), int(child.get("depth_used", depth))),
        "state_key": str(trans.get("state_key", trans.get("requested_state_key", "NA"))),
    }
    cache[key] = out
    return out


def decide_retransmission(
    current_attempt: int,
    max_attempts: int,
    base_link_success_prob: float,
    distance_m: float,
    attempt_spacing_ms: float,
    impairment_b: float,
    extra_ms: float,
    exp_scale_ms: float,
    tail_model: str,
    base_cong_delay_ms: float,
    base_cbr: float,
    base_p_col: float,
    cfg: RetxPolicyConfig,
    scenario: str = "UrbMask",
) -> Dict[str, Any]:
    policy = (cfg.policy or "classical").strip().lower()
    cur = int(current_attempt)
    maxa = int(max_attempts)
    if cur >= maxa:
        return {
            "decision": False,
            "policy": policy,
            "reason": "NO_BUDGET",
            "current_est_delay_ms": np.nan,
            "predicted_next_delay_ms": np.nan,
            "incremental_delay_ms": np.nan,
            "predicted_utility": 0.0,
            "expected_gain": 0.0,
            "cost_ci": np.nan,
            "gain_over_cost": np.nan,
            "predicted_timely": False,
            "score": np.nan,
            "slack_ms": np.nan,
            "success_prob_next": float(base_link_success_prob),
            "next_attempt_idx": cur + 1,
            "predicted_next_cbr": np.nan,
            "predicted_next_p_col": np.nan,
            "predicted_next_cong_delay_ms": np.nan,
            "predicted_busy_pressure": np.nan,
            "effective_cost_mode": str((cfg.cost_mode or "delay_only").lower()),
            "delay_norm": np.nan,
            "airtime_norm": np.nan,
            "delay_term": np.nan,
            "airtime_term": np.nan,
            "resource_term": np.nan,
            "cost_multiplier": np.nan,
            "horizon_ms": np.nan,
            "chain_expected_gain": 0.0,
            "chain_expected_cost": 0.0,
            "chain_gain_over_cost": np.nan,
            "chain_score": np.nan,
            "chain_best_pred_delay_ms": np.nan,
            "chain_best_pred_utility": 0.0,
            "mdp_model_hit": False,
            "mdp_model_samples": 0,
            "mdp_lookup_level": "NO_BUDGET",
            "mdp_q_continue": np.nan,
            "mdp_q_stop": 0.0,
            "mdp_value": np.nan,
            "mdp_depth_used": 0,
            "mdp_cost_scale": float(_nonnegative_finite(cfg.mdp_cost_scale, default=1.0)),
            "mdp_model_miss": False,
            "mdp_chain_fallback_used": False,
            "mdp_chain_score_used": np.nan,
            "mdp_decision_source": "no_budget",
            "mdp_cost_raw": np.nan,
            "mdp_cost_scaled": np.nan,
            "mdp_expected_success_term": np.nan,
            "mdp_future_fail_term": np.nan,
            "mdp_exact_hit": False,
            "mdp_coarse_hit": False,
            "mdp_global_default_hit": False,
            "mdp_hit_kind": "NO_BUDGET",
            "mdp_effective_min_samples": 0,
            "mdp_requested_state_key": "NA",
            "mdp_lookup_scope": "NA",
            "mdp_lookup_rank": -1,
            "mdp_raw_margin": np.nan,
            "mdp_threshold_applied": np.nan,
            "mdp_thresholded_margin": np.nan,
        }

    next_attempt = cur + 1
    pred_cur_delay = estimate_attempt_delay_ms(
        distance_m=float(distance_m),
        attempt_idx=int(cur),
        attempt_spacing_ms=float(attempt_spacing_ms),
        impairment_b=float(impairment_b),
        extra_ms=float(extra_ms),
        exp_scale_ms=float(exp_scale_ms),
        tail_model=str(tail_model),
        cong_delay_ms=float(
            estimate_attempt_conditions(
                attempt_idx=int(cur),
                base_link_success_prob=float(base_link_success_prob),
                base_cbr=float(base_cbr),
                base_p_col=float(base_p_col),
                base_cong_delay_ms=float(base_cong_delay_ms),
                cfg=cfg,
            )["cong_delay_ms_eff"]
        ),
    )
    next_cond = estimate_attempt_conditions(
        attempt_idx=int(next_attempt),
        base_link_success_prob=float(base_link_success_prob),
        base_cbr=float(base_cbr),
        base_p_col=float(base_p_col),
        base_cong_delay_ms=float(base_cong_delay_ms),
        cfg=cfg,
    )
    pred_next_delay = estimate_attempt_delay_ms(
        distance_m=float(distance_m),
        attempt_idx=int(next_attempt),
        attempt_spacing_ms=float(attempt_spacing_ms),
        impairment_b=float(impairment_b),
        extra_ms=float(extra_ms),
        exp_scale_ms=float(exp_scale_ms),
        tail_model=str(tail_model),
        cong_delay_ms=float(next_cond["cong_delay_ms_eff"]),
    )
    incremental_delay = max(0.0, float(pred_next_delay - pred_cur_delay))
    util = utility_from_delay_ms(
        delay_ms=float(pred_next_delay),
        deadline_ms=float(cfg.deadline_ms),
        grace_ms=float(cfg.grace_ms),
        beta=float(cfg.beta),
        gamma=float(cfg.gamma),
    )
    timely = bool(pred_next_delay <= float(cfg.deadline_ms))
    cost_info = compute_cost_details(
        cost_mode=str(cfg.cost_mode),
        incremental_delay_ms=float(incremental_delay),
        deadline_ms=float(cfg.deadline_ms),
        grace_ms=float(cfg.grace_ms),
        cbr=float(next_cond["cbr_eff"]),
        p_col=float(next_cond["p_col_eff"]),
        airtime_ms=float(next_cond["airtime_ms"]),
        congestion_knee=float(cfg.congestion_knee),
        cost_airtime_weight=float(cfg.cost_airtime_weight),
        cost_resource_weight=float(cfg.cost_resource_weight),
    )
    cost = float(cost_info["cost_ci"])
    expected_gain = float(next_cond["p_success_eff"] * util)
    gain_over_cost = float(expected_gain / max(cost, 1e-9))
    slack_ms = float(cfg.deadline_ms - pred_next_delay)

    mdp: Dict[str, Any] = {}
    mdp_decision_source = "not_applicable"
    mdp_chain_score_used = np.nan

    chain = _expected_chain_metrics(
        current_attempt=int(cur),
        max_attempts=int(maxa),
        base_link_success_prob=float(base_link_success_prob),
        distance_m=float(distance_m),
        attempt_spacing_ms=float(attempt_spacing_ms),
        impairment_b=float(impairment_b),
        extra_ms=float(extra_ms),
        exp_scale_ms=float(exp_scale_ms),
        tail_model=str(tail_model),
        base_cong_delay_ms=float(base_cong_delay_ms),
        base_cbr=float(base_cbr),
        base_p_col=float(base_p_col),
        cfg=cfg,
        pred_cur_delay=float(pred_cur_delay),
    )

    if policy == "classical":
        decision = True
        reason = "CLASSICAL_ALWAYS_RETX"
        score = np.nan
        mdp_decision_source = "classical"
    elif policy == "nomikos":
        decision = timely
        reason = "NOMIKOS_TIMELY_OK" if decision else "NOMIKOS_PREDICTED_LATE"
        score = float(next_cond["p_success_eff"] if timely else 0.0)
        mdp_decision_source = "nomikos"
    elif policy == "udrc":
        score = float(chain["chain_score"])
        decision = bool(score > 0.0)
        reason = "UDRC_CHAIN_GAIN_GT_COST" if decision else "UDRC_CHAIN_GAIN_LE_COST"
        mdp_decision_source = "udrc"
    elif policy in ("mdp", "mdp_lite", "mdplite"):
        model = cfg.mdp_model_obj if getattr(cfg, 'mdp_model_obj', None) is not None else (load_mdp_lite_model(cfg.mdp_model_path) if str(cfg.mdp_model_path or "").strip() else None)
        mdp = _mdp_lite_value(
            scenario=str(scenario),
            remaining_budget=int(maxa - cur),
            distance_m=float(distance_m),
            current_delay_ms=float(pred_cur_delay),
            cbr=float(base_cbr),
            cfg=cfg,
            model=model,
            cache={},
            depth=0,
        )
        if bool(mdp.get("hit", False)):
            q_continue = float(mdp.get("q_continue", np.nan))
            q_stop = float(mdp.get("q_stop", 0.0))
            raw_margin = q_continue - q_stop
            thresholded_margin = raw_margin - float(cfg.mdp_threshold)
            score = float(thresholded_margin)
            decision = bool(thresholded_margin > 0.0)
            reason = "MDPLITE_MARGIN_GT_THRESHOLD" if decision else "MDPLITE_MARGIN_LE_THRESHOLD"
            mdp["raw_margin"] = float(raw_margin)
            mdp["thresholded_margin"] = float(thresholded_margin)
            mdp["threshold_applied"] = float(cfg.mdp_threshold)
            mdp_decision_source = "model"
        else:
            score = float(chain["chain_score"])
            mdp_chain_score_used = float(score)
            if bool(cfg.mdp_fallback_to_chain):
                raw_margin = float(score)
                thresholded_margin = float(raw_margin - float(cfg.mdp_threshold))
                decision = bool(thresholded_margin > 0.0)
                reason = "MDPLITE_CHAINFALLBACK_GT_THRESHOLD" if decision else "MDPLITE_CHAINFALLBACK_LE_THRESHOLD"
                mdp["raw_margin"] = float(score)
                mdp["thresholded_margin"] = float(thresholded_margin)
                mdp["threshold_applied"] = float(cfg.mdp_threshold)
                mdp["hit_kind"] = "chain_fallback"
                mdp["lookup_level"] = "chain_fallback"
                mdp["lookup_scope"] = "chain_fallback"
                mdp["lookup_rank"] = -1
                mdp_decision_source = "chain_fallback"
            else:
                decision = False
                reason = "MDPLITE_NO_MODEL_DROP"
                mdp["raw_margin"] = float('nan')
                mdp["thresholded_margin"] = float('nan')
                mdp["threshold_applied"] = float(cfg.mdp_threshold)
                mdp["hit_kind"] = "missing"
                mdp["lookup_level"] = "missing"
                mdp["lookup_scope"] = "none"
                mdp["lookup_rank"] = -1
                mdp_decision_source = "drop"
    elif policy in ("no_retx", "noretx", "none"):
        decision = False
        reason = "NO_RETX_POLICY"
        score = np.nan
        mdp_decision_source = "noretx"
    else:
        raise ValueError(f"Unsupported retx policy: {cfg.policy}")

    return {
        "decision": bool(decision),
        "policy": policy,
        "reason": str(reason),
        "current_est_delay_ms": float(pred_cur_delay),
        "predicted_next_delay_ms": float(pred_next_delay),
        "incremental_delay_ms": float(incremental_delay),
        "predicted_utility": float(util),
        "expected_gain": float(expected_gain),
        "cost_ci": float(cost),
        "gain_over_cost": float(gain_over_cost),
        "predicted_timely": bool(timely),
        "score": float(score) if np.isfinite(score) else np.nan,
        "slack_ms": float(slack_ms),
        "success_prob_next": float(next_cond["p_success_eff"]),
        "next_attempt_idx": int(next_attempt),
        "predicted_next_cbr": float(next_cond["cbr_eff"]),
        "predicted_next_p_col": float(next_cond["p_col_eff"]),
        "predicted_next_cong_delay_ms": float(next_cond["cong_delay_ms_eff"]),
        "predicted_busy_pressure": float(next_cond["busy_pressure"]),
        "effective_cost_mode": str(cost_info["effective_cost_mode"]),
        "delay_norm": float(cost_info["delay_norm"]),
        "airtime_norm": float(cost_info["airtime_norm"]),
        "delay_term": float(cost_info.get("delay_term", np.nan)) if np.isfinite(cost_info.get("delay_term", np.nan)) else np.nan,
        "airtime_term": float(cost_info.get("airtime_term", np.nan)) if np.isfinite(cost_info.get("airtime_term", np.nan)) else np.nan,
        "resource_term": float(cost_info["resource_term"]),
        "cost_multiplier": float(cost_info["cost_multiplier"]),
        "horizon_ms": float(cost_info["horizon_ms"]),
        "chain_expected_gain": float(chain["chain_expected_gain"]),
        "chain_expected_cost": float(chain["chain_expected_cost"]),
        "chain_gain_over_cost": float(chain["chain_gain_over_cost"]) if np.isfinite(chain["chain_gain_over_cost"]) else np.nan,
        "chain_score": float(chain["chain_score"]) if np.isfinite(chain["chain_score"]) else np.nan,
        "chain_best_pred_delay_ms": float(chain["chain_best_pred_delay_ms"]) if np.isfinite(chain["chain_best_pred_delay_ms"]) else np.nan,
        "chain_best_pred_utility": float(chain["chain_best_pred_utility"]),
        "mdp_model_hit": bool(mdp.get("hit", False)) if policy in ("mdp", "mdp_lite", "mdplite") else False,
        "mdp_model_miss": (policy in ("mdp", "mdp_lite", "mdplite")) and (not bool(mdp.get("hit", False))),
        "mdp_exact_hit": bool(str(mdp.get("lookup_level", "")).lower() in ("exact", "global_exact")) if policy in ("mdp", "mdp_lite", "mdplite") and bool(mdp.get("hit", False)) else False,
        "mdp_coarse_hit": bool(str(mdp.get("hit_kind", "")).lower() == "coarse") if policy in ("mdp", "mdp_lite", "mdplite") and bool(mdp.get("hit", False)) else False,
        "mdp_global_default_hit": bool(str(mdp.get("hit_kind", "")).lower() == "global_default") if policy in ("mdp", "mdp_lite", "mdplite") and bool(mdp.get("hit", False)) else False,
        "mdp_hit_kind": str(mdp.get("hit_kind", "NA")) if policy in ("mdp", "mdp_lite", "mdplite") else "NA",
        "mdp_model_samples": int(mdp.get("samples", 0)) if policy in ("mdp", "mdp_lite", "mdplite") else 0,
        "mdp_effective_min_samples": int(mdp.get("effective_min_samples", 0)) if policy in ("mdp", "mdp_lite", "mdplite") else 0,
        "mdp_lookup_level": str(mdp.get("lookup_level", "NA")) if policy in ("mdp", "mdp_lite", "mdplite") else "NA",
        "mdp_state_key": str(mdp.get("state_key", "NA")) if policy in ("mdp", "mdp_lite", "mdplite") else "NA",
        "mdp_requested_state_key": str(mdp.get("requested_state_key", "NA")) if policy in ("mdp", "mdp_lite", "mdplite") else "NA",
        "mdp_requested_signature": str(mdp.get("requested_signature", {})) if policy in ("mdp", "mdp_lite", "mdplite") else "{}",
        "mdp_chain_fallback_used": bool(mdp_decision_source == "chain_fallback") if policy in ("mdp", "mdp_lite", "mdplite") else False,
        "mdp_fallback_used": bool(
            (mdp_decision_source == "chain_fallback")
            or (
                bool(mdp.get("hit", False))
                and str(mdp.get("hit_kind", "exact")).lower() not in ("exact",)
            )
        ) if policy in ("mdp", "mdp_lite", "mdplite") else False,
        "mdp_lookup_scope": str(mdp.get("lookup_scope", "NA")) if policy in ("mdp", "mdp_lite", "mdplite") else "NA",
        "mdp_lookup_rank": int(mdp.get("lookup_rank", -1)) if policy in ("mdp", "mdp_lite", "mdplite") else -1,
        "mdp_decision_source": str(mdp_decision_source) if policy in ("mdp", "mdp_lite", "mdplite") else str(mdp_decision_source),
        "mdp_chain_score_used": float(mdp_chain_score_used) if policy in ("mdp", "mdp_lite", "mdplite") and np.isfinite(mdp_chain_score_used) else np.nan,
        "mdp_q_continue": float(mdp.get("q_continue", np.nan)) if policy in ("mdp", "mdp_lite", "mdplite") and np.isfinite(mdp.get("q_continue", np.nan)) else np.nan,
        "mdp_q_stop": float(mdp.get("q_stop", 0.0)) if policy in ("mdp", "mdp_lite", "mdplite") else 0.0,
        "mdp_cost_scale": float(mdp.get("cost_scale", cfg.mdp_cost_scale)) if policy in ("mdp", "mdp_lite", "mdplite") else np.nan,
        "mdp_cost_raw": float(mdp.get("continue_cost_ci", np.nan)) if policy in ("mdp", "mdp_lite", "mdplite") and np.isfinite(mdp.get("continue_cost_ci", np.nan)) else np.nan,
        "mdp_cost_scaled": float(mdp.get("continue_cost_ci_scaled", np.nan)) if policy in ("mdp", "mdp_lite", "mdplite") and np.isfinite(mdp.get("continue_cost_ci_scaled", np.nan)) else np.nan,
        "mdp_expected_success_term": float(mdp.get("success_term", np.nan)) if policy in ("mdp", "mdp_lite", "mdplite") and np.isfinite(mdp.get("success_term", np.nan)) else np.nan,
        "mdp_future_fail_term": float(mdp.get("future_fail_term", np.nan)) if policy in ("mdp", "mdp_lite", "mdplite") and np.isfinite(mdp.get("future_fail_term", np.nan)) else np.nan,
        "mdp_raw_margin": float(mdp.get("raw_margin", np.nan)) if policy in ("mdp", "mdp_lite", "mdplite") and np.isfinite(mdp.get("raw_margin", np.nan)) else np.nan,
        "mdp_threshold_applied": float(mdp.get("threshold_applied", np.nan)) if policy in ("mdp", "mdp_lite", "mdplite") and np.isfinite(mdp.get("threshold_applied", np.nan)) else np.nan,
        "mdp_thresholded_margin": float(mdp.get("thresholded_margin", np.nan)) if policy in ("mdp", "mdp_lite", "mdplite") and np.isfinite(mdp.get("thresholded_margin", np.nan)) else np.nan,
        "mdp_value": float(mdp.get("value", np.nan)) if policy in ("mdp", "mdp_lite", "mdplite") and np.isfinite(mdp.get("value", np.nan)) else np.nan,
        "mdp_depth_used": int(mdp.get("depth_used", 0)) if policy in ("mdp", "mdp_lite", "mdplite") else 0,
    }
