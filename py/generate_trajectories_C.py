from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from paths_C import ensure_run_dirs_a, make_run_id, load_latest_run_id, default_run_prefix
from run_logging import log_command, update_manifest, snapshot_file

from modules.road_geometry import RefPlusGeometry
from modules.traffic_signals import SignalPlan
from modules.traffic_idm import IDMParams, idm_accel
from modules.vehicle_profiles import sample_vehicle_profile


def _pick_run_id(arg_run_id: str) -> str:
    s = (arg_run_id or "").strip()
    if s == "":
        return make_run_id(prefix=default_run_prefix())
    if s.lower() == "latest":
        rid = load_latest_run_id()
        return rid if rid else make_run_id(prefix=default_run_prefix())
    return s


def _ensure_time_key(df: pd.DataFrame) -> pd.DataFrame:
    if "time_key" in df.columns:
        return df
    df = df.copy()
    if "time_s" in df.columns:
        df["time_key"] = df["time_s"].round(3)
    else:
        raise ValueError("trajectory must have time_key or time_s")
    return df


def _spawn_ok(vlist: list[dict], min_spawn_gap_m: float, *, kind: str, s_spawn: float) -> bool:
    """
    Spawn gap check using lane-coordinate (stable, no need x_m/y_m in state dict).

    kind="main": s starts at 0 and increases; allow spawn if closest vehicle to start is far enough:
        min(s) >= min_spawn_gap_m

    kind="cross": s_cross starts at cross_half_length and decreases toward 0; allow spawn if
        (s_spawn - max(s_cross)) >= min_spawn_gap_m
    """
    if not vlist:
        return True

    g = float(min_spawn_gap_m)

    if kind == "main":
        s_min = min(float(v.get("s", 1e9)) for v in vlist)
        return s_min >= g

    if kind == "cross":
        s_max = max(float(v.get("s_cross", -1e9)) for v in vlist)
        return (float(s_spawn) - s_max) >= g

    # turn or others: do not block
    return True


def _road_tag_main(direction: int, lane_id: int) -> str:
    # MAIN_DIR+1_Ln / MAIN_DIR-1_Ln
    return f"MAIN_D{int(direction):+d}_L{int(lane_id)}"


def _road_tag_cross(which: int, direction: int, lane_id: int) -> str:
    # CROSS_I1_D+1_L0 / CROSS_I2_D-1_L0
    return f"CROSS_I{int(which)}_D{int(direction):+d}_L{int(lane_id)}"


def _road_tag_turn(which: int, turn_kind: str, lane_id: int) -> str:
    # TURN_I1_R_L0 / TURN_I2_L_L0
    return f"TURN_I{int(which)}_{turn_kind}_L{int(lane_id)}"


def _lane_key(kind: str, which: int, direction: int, lane_id: int) -> tuple[str, int, int, int]:
    # kind: "main" / "cross" / "turn"
    return (str(kind), int(which), int(direction), int(lane_id))


def _init_lane_state(
    geom: RefPlusGeometry,
    n_lanes_per_dir: int,
    cross_enable: bool,
) -> tuple[list[tuple[str, int, int, int]], dict[tuple[str, int, int, int], list[dict]]]:
    lanes: list[tuple[str, int, int, int]] = []
    lane_veh: dict[tuple[str, int, int, int], list[dict]] = {}

    # main lanes
    for d in (+1, -1):
        for lane in range(int(n_lanes_per_dir)):
            k = _lane_key("main", 0, d, lane)
            lanes.append(k)
            lane_veh[k] = []

    if cross_enable:
        # cross lanes (1 lane each direction for each intersection)
        for which in (1, 2):
            for d in (+1, -1):
                k = _lane_key("cross", which, d, 0)
                lanes.append(k)
                lane_veh[k] = []

        # turning lanes: one lane each for each intersection, left/right
        for which in (1, 2):
            k = _lane_key("turn", which, 0, 0)
            lanes.append(k)
            lane_veh[k] = []

    return lanes, lane_veh


def _idm_params_from_vehicle(vv: dict, default_params: IDMParams) -> IDMParams:
    return IDMParams(
        v0_mps=float(vv.get("idm_v0_mps", default_params.v0_mps)),
        T_s=float(vv.get("idm_T_s", default_params.T_s)),
        a_mps2=float(vv.get("idm_a_mps2", default_params.a_mps2)),
        b_mps2=float(vv.get("idm_b_mps2", default_params.b_mps2)),
        s0_m=float(vv.get("idm_s0_m", default_params.s0_m)),
        delta=float(vv.get("idm_delta", default_params.delta)),
    )


def _simulate_refplus_idm(
    geom: RefPlusGeometry,
    plan_i1: SignalPlan,
    plan_i2: SignalPlan,
    flow_main_vph: float,
    flow_cross_vph: float,
    p_turn_i1: float,
    p_turn_i2: float,
    p_right: float,
    p_left: float,
    veh_length_m: float = 4.5,
    min_spawn_gap_m: float = 10.0,
    duration_s: float = 60.0,
    dt_s: float = 0.1,
    seed: int = 1,
    enable_signals: bool = True,
    cross_enable: bool = True,
    cross_half_length_m: float = 300.0,
    idm_params: IDMParams | None = None,
    enable_vehicle_mix: bool = False,
    car_ratio: float = 0.85,
    heavy_length_m: float = 11.5,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))

    if idm_params is None:
        idm_params = IDMParams()

    # veh id counter
    next_vid = 1

    # lanes state
    lanes, lane_veh = _init_lane_state(geom, geom.n_lanes_per_dir, cross_enable)

    # spawning clocks
    flow_main = float(flow_main_vph) / 3600.0
    flow_cross = float(flow_cross_vph) / 3600.0
    next_spawn_t: dict[tuple[str, int, int, int], float] = {k: 0.0 for k in lanes}

    # precompute times
    times = np.arange(0.0, float(duration_s) + 1e-9, float(dt_s))

    records = []

    def main_x_from_s(s: float, direction: int) -> float:
        # direction +1: x = s (increasing), direction -1: x = L - s
        L = float(geom.road_length_m)
        return float(s) if int(direction) >= 0 else float(L - s)

    # stop line positions (x coordinates)
    stop_i1_x = float(geom.i1_x)
    stop_i2_x = float(geom.i2_x)

    # for cross road: x = i1 or i2, y moves along, limited by cross_half_length_m
    def cross_xy(which: int, s_cross: float, direction: int) -> tuple[float, float]:
        x = float(geom.i1_x) if int(which) == 1 else float(geom.i2_x)
        # direction +1: y increases; direction -1: y decreases
        y = float(s_cross) if int(direction) >= 0 else float(-s_cross)
        # cross road is centered at y=0
        return x, y

    # turn path helper (simple bezier via geom)
    def turn_xy(which: int, turn_kind: str, u: float) -> tuple[float, float]:
        # u in [0,1]
        return geom.turn_xy(which=int(which), turn_kind=str(turn_kind), u=float(u))

    for t in times:
        t = float(t)

        # signal states per intersection
        main_green_i1 = (not enable_signals) or plan_i1.main_is_green(t)
        main_green_i2 = (not enable_signals) or plan_i2.main_is_green(t)
        # cross green is defined by the signal plan phase (MAIN_GREEN / ALL_RED / CROSS_GREEN)
        cross_green_i1 = (not enable_signals) or plan_i1.cross_is_green(t)
        cross_green_i2 = (not enable_signals) or plan_i2.cross_is_green(t)

        # --- update lane vehicles ---
        for k in lanes:
            kind, which, d, lane = k

            # spawning
            if kind == "main" and flow_main > 1e-9:
                while t >= next_spawn_t[k]:
                    vlist = lane_veh[k]
                    if len(vlist) > 0:
                        # new vehicle at start of road for direction
                        s_new = 0.0
                        x_new = main_x_from_s(s_new, d)
                        y_new = float(geom.lane_center_y(x_new, d, lane))
                        if _spawn_ok(vlist, min_spawn_gap_m, kind="main", s_spawn=0.0):
                            vid = next_vid
                            next_vid += 1
                            prof = sample_vehicle_profile(
                                rng,
                                idm_params,
                                base_length_m=float(veh_length_m),
                                enable_vehicle_mix=bool(enable_vehicle_mix),
                                car_ratio=float(car_ratio),
                                heavy_length_m=float(heavy_length_m),
                            )
                            v = {
                                "veh_id": vid,
                                "kind": "main",
                                "which": 0,
                                "direction": int(d),
                                "lane": int(lane),
                                "s": float(s_new),
                                "v": float(prof.idm_params.v0_mps * 0.7),
                                "a": 0.0,
                                "road_tag": _road_tag_main(d, lane),
                                "veh_type": str(prof.veh_type),
                                "veh_length_m": float(prof.veh_length_m),
                                "idm_v0_mps": float(prof.idm_params.v0_mps),
                                "idm_T_s": float(prof.idm_params.T_s),
                                "idm_a_mps2": float(prof.idm_params.a_mps2),
                                "idm_b_mps2": float(prof.idm_params.b_mps2),
                                "idm_s0_m": float(prof.idm_params.s0_m),
                                "idm_delta": float(prof.idm_params.delta),
                            }
                            vlist.append(v)
                            next_spawn_t[k] += float(rng.exponential(1.0 / flow_main))
                        else:
                            break
                    else:
                        s_new = 0.0
                        x_new = main_x_from_s(s_new, d)
                        y_new = float(geom.lane_center_y(x_new, d, lane))
                        vid = next_vid
                        next_vid += 1
                        prof = sample_vehicle_profile(
                            rng,
                            idm_params,
                            base_length_m=float(veh_length_m),
                            enable_vehicle_mix=bool(enable_vehicle_mix),
                            car_ratio=float(car_ratio),
                            heavy_length_m=float(heavy_length_m),
                        )
                        v = {
                            "veh_id": vid,
                            "kind": "main",
                            "which": 0,
                            "direction": int(d),
                            "lane": int(lane),
                            "s": float(s_new),
                            "v": float(prof.idm_params.v0_mps * 0.7),
                            "a": 0.0,
                            "road_tag": _road_tag_main(d, lane),
                            "veh_type": str(prof.veh_type),
                            "veh_length_m": float(prof.veh_length_m),
                            "idm_v0_mps": float(prof.idm_params.v0_mps),
                            "idm_T_s": float(prof.idm_params.T_s),
                            "idm_a_mps2": float(prof.idm_params.a_mps2),
                            "idm_b_mps2": float(prof.idm_params.b_mps2),
                            "idm_s0_m": float(prof.idm_params.s0_m),
                            "idm_delta": float(prof.idm_params.delta),
                        }
                        vlist.append(v)
                        next_spawn_t[k] += float(rng.exponential(1.0 / flow_main))

            if kind == "cross" and cross_enable and flow_cross > 1e-9:
                while t >= next_spawn_t[k]:
                    vlist = lane_veh[k]
                    s_new = float(cross_half_length_m)  # start from far end to center
                    # direction +1: from negative to center -> we use s decreasing? keep consistent with update below
                    # We'll store s_cross as distance from center, positive.
                    x_new, y_new = cross_xy(which, s_new, d)
                    if not vlist or _spawn_ok(vlist, min_spawn_gap_m, kind="cross", s_spawn=s_new):
                        vid = next_vid
                        next_vid += 1
                        prof = sample_vehicle_profile(
                            rng,
                            idm_params,
                            base_length_m=float(veh_length_m),
                            enable_vehicle_mix=bool(enable_vehicle_mix),
                            car_ratio=float(car_ratio),
                            heavy_length_m=float(heavy_length_m),
                        )
                        v = {
                            "veh_id": vid,
                            "kind": "cross",
                            "which": int(which),
                            "direction": int(d),
                            "lane": int(lane),
                            "s_cross": float(s_new),
                            "v": float(prof.idm_params.v0_mps * 0.6),
                            "a": 0.0,
                            "road_tag": _road_tag_cross(which, d, lane),
                            "turn": False,
                            "turn_kind": "",
                            "turn_u": 0.0,
                            "veh_type": str(prof.veh_type),
                            "veh_length_m": float(prof.veh_length_m),
                            "idm_v0_mps": float(prof.idm_params.v0_mps),
                            "idm_T_s": float(prof.idm_params.T_s),
                            "idm_a_mps2": float(prof.idm_params.a_mps2),
                            "idm_b_mps2": float(prof.idm_params.b_mps2),
                            "idm_s0_m": float(prof.idm_params.s0_m),
                            "idm_delta": float(prof.idm_params.delta),
                        }
                        # decide turning when approaching intersection center
                        p_turn = float(p_turn_i1) if int(which) == 1 else float(p_turn_i2)
                        if rng.random() < p_turn:
                            v["turn"] = True
                            if rng.random() < float(p_right):
                                v["turn_kind"] = "R"
                            else:
                                v["turn_kind"] = "L"
                        vlist.append(v)
                        next_spawn_t[k] += float(rng.exponential(1.0 / flow_cross))
                    else:
                        break

            # dynamics update
            vlist = lane_veh[k]
            if not vlist:
                continue

            # sort by longitudinal position along lane for car-following
            # For main: sort by s, increasing (front has larger s)
            # For cross: sort by s_cross decreasing towards 0? We'll treat "progress" as (cross_half_length - s_cross)
            if kind == "main":
                vlist.sort(key=lambda vv: float(vv["s"]))
            elif kind == "cross":
                vlist.sort(key=lambda vv: float(vv["s_cross"]), reverse=True)

            # update each vehicle
            new_list = []
            for idx, vv in enumerate(vlist):
                if kind == "main":
                    s = float(vv["s"])
                    v = float(vv["v"])
                    veh_len = float(vv.get("veh_length_m", veh_length_m))
                    idm_local = _idm_params_from_vehicle(vv, idm_params)
                    # leader gap
                    if idx == len(vlist) - 1:
                        s_lead = None
                        v_lead = None
                        gap = float("inf")
                    else:
                        s_lead = float(vlist[idx + 1]["s"])
                        v_lead = float(vlist[idx + 1]["v"])
                        gap = max(0.1, (s_lead - s) - float(veh_len))

                    # signal constraint: stop at i1 / i2 when red
                    # direction +1: increasing x -> increasing s
                    # direction -1: decreasing x -> increasing s still (since x = L - s)
                    x = main_x_from_s(s, d)

                    # choose which intersection we are approaching (based on x)
                    # use local stop_x per direction
                    # We treat red as a virtual leader at stop line if within zone ahead.
                    # intersection zone as in geometry
                    zone = float(geom.intersection_zone_m)
                    a_sig = None
                    if int(d) >= 0:
                        # approaching i1 then i2
                        if (x < stop_i1_x) and (stop_i1_x - x) <= zone and (not main_green_i1):
                            # virtual leader at stop_i1_x
                            s_stop = float(stop_i1_x)
                            gap_sig = max(0.1, (s_stop - s) - float(veh_len))
                            a_sig = idm_accel(v, 0.0, gap_sig, idm_local)
                        elif (x < stop_i2_x) and (stop_i2_x - x) <= zone and (not main_green_i2):
                            s_stop = float(stop_i2_x)
                            gap_sig = max(0.1, (s_stop - s) - float(veh_len))
                            a_sig = idm_accel(v, 0.0, gap_sig, idm_local)
                    else:
                        # direction -1: x decreasing as s increases; stop lines in x
                        if (x > stop_i2_x) and (x - stop_i2_x) <= zone and (not main_green_i2):
                            # virtual leader at stop_i2_x
                            # convert x_stop to s_stop: x_stop = L - s_stop -> s_stop = L - x_stop
                            s_stop = float(geom.road_length_m - stop_i2_x)
                            gap_sig = max(0.1, (s_stop - s) - float(veh_len))
                            a_sig = idm_accel(v, 0.0, gap_sig, idm_local)
                        elif (x > stop_i1_x) and (x - stop_i1_x) <= zone and (not main_green_i1):
                            s_stop = float(geom.road_length_m - stop_i1_x)
                            gap_sig = max(0.1, (s_stop - s) - float(veh_len))
                            a_sig = idm_accel(v, 0.0, gap_sig, idm_local)

                    # normal IDM
                    if s_lead is None:
                        a_idm = idm_accel(v, v, gap, idm_local)
                    else:
                        a_idm = idm_accel(v, v_lead, gap, idm_local)

                    a = float(a_idm)
                    if a_sig is not None:
                        a = min(a, float(a_sig))

                    # integrate
                    v = max(0.0, v + a * float(dt_s))
                    s = s + v * float(dt_s)

                    # termination
                    if s <= float(geom.road_length_m) + 50.0:
                        vv["s"] = s
                        vv["v"] = v
                        vv["a"] = a
                        new_list.append(vv)

                        x = main_x_from_s(s, d)
                        y = float(geom.lane_center_y(x, d, lane))
                        records.append(
                            {
                                "time_s": t,
                                "time_key": round(t, 3),
                                "veh_id": int(vv["veh_id"]),
                                "x_m": float(x),
                                "y_m": float(y),
                                "speed_mps": float(v),
                                "road_tag": str(vv["road_tag"]),
                                "veh_type": str(vv.get("veh_type", "car")),
                                "veh_length_m": float(vv.get("veh_length_m", veh_length_m)),
                                "idm_v0_mps": float(vv.get("idm_v0_mps", idm_params.v0_mps)),
                                "idm_T_s": float(vv.get("idm_T_s", idm_params.T_s)),
                            }
                        )

                elif kind == "cross":
                    s_cross = float(vv["s_cross"])
                    v = float(vv["v"])
                    veh_len = float(vv.get("veh_length_m", veh_length_m))
                    idm_local = _idm_params_from_vehicle(vv, idm_params)

                    # cross signal check (which intersection)
                    if int(which) == 1:
                        allow = bool(cross_green_i1)
                    else:
                        allow = bool(cross_green_i2)

                    # leader gap: toward center (s_cross decreasing to 0)
                    if idx == len(vlist) - 1:
                        gap = float("inf")
                        v_lead = 0.0
                    else:
                        s_lead = float(vlist[idx + 1]["s_cross"])
                        v_lead = float(vlist[idx + 1]["v"])
                        # since sorted descending, leader has smaller s_cross (closer to center)
                        gap = max(0.1, (s_cross - s_lead) - float(veh_len))

                    # stopping at center when red: use virtual leader at s_cross=0
                    a_sig = None
                    zone = float(geom.intersection_zone_m)
                    if (not allow) and (s_cross <= zone):
                        gap_sig = max(0.1, s_cross - float(veh_len))
                        a_sig = idm_accel(v, 0.0, gap_sig, idm_local)

                    # normal IDM
                    a_idm = idm_accel(v, v_lead, gap, idm_local)
                    a = float(a_idm)
                    if a_sig is not None:
                        a = min(a, float(a_sig))

                    # integrate: move toward center (s_cross decreases)
                    v = max(0.0, v + a * float(dt_s))
                    s_cross = max(0.0, s_cross - v * float(dt_s))

                    # turn decision when reached center and allowed
                    if vv.get("turn", False) and allow and s_cross <= 1.0:
                        # start turn trajectory
                        vv["kind"] = "turn"
                        vv["turn_u"] = 0.0
                        # tag update
                        turn_kind = str(vv.get("turn_kind", "R"))
                        vv["road_tag"] = _road_tag_turn(which, turn_kind, 0)
                        # we drop from cross list; the same dict will later be picked up by "turn" lane key
                        # easiest: just store it into a dedicated lane key
                        turn_key = _lane_key("turn", which, 0, 0)
                        lane_veh[turn_key].append(vv)
                        continue

                    # termination: out of cross range
                    if s_cross >= 0.0:
                        vv["s_cross"] = s_cross
                        vv["v"] = v
                        vv["a"] = a
                        new_list.append(vv)

                        x, y = cross_xy(which, s_cross, d)
                        records.append(
                            {
                                "time_s": t,
                                "time_key": round(t, 3),
                                "veh_id": int(vv["veh_id"]),
                                "x_m": float(x),
                                "y_m": float(y),
                                "speed_mps": float(v),
                                "road_tag": str(vv["road_tag"]),
                                "veh_type": str(vv.get("veh_type", "car")),
                                "veh_length_m": float(vv.get("veh_length_m", veh_length_m)),
                                "idm_v0_mps": float(vv.get("idm_v0_mps", idm_params.v0_mps)),
                                "idm_T_s": float(vv.get("idm_T_s", idm_params.T_s)),
                            }
                        )

                elif kind == "turn":
                    # turn vehicles are stored under ("turn", which, 0, 0)
                    u = float(vv.get("turn_u", 0.0))
                    v = float(vv["v"])
                    turn_kind = str(vv.get("turn_kind", "R"))

                    # simple constant speed along curve (lightweight)
                    u = u + (v * float(dt_s)) / max(1e-6, float(geom.turn_path_length_m(which, turn_kind)))
                    if u >= 1.0:
                        continue

                    vv["turn_u"] = u
                    vv["a"] = 0.0

                    x, y = turn_xy(which, turn_kind, u)
                    records.append(
                        {
                            "time_s": t,
                            "time_key": round(t, 3),
                            "veh_id": int(vv["veh_id"]),
                            "x_m": float(x),
                            "y_m": float(y),
                            "speed_mps": float(v),
                            "road_tag": str(vv["road_tag"]),
                            "veh_type": str(vv.get("veh_type", "car")),
                            "veh_length_m": float(vv.get("veh_length_m", veh_length_m)),
                            "idm_v0_mps": float(vv.get("idm_v0_mps", idm_params.v0_mps)),
                            "idm_T_s": float(vv.get("idm_T_s", idm_params.T_s)),
                        }
                    )
                    new_list.append(vv)

            lane_veh[k] = new_list

    df = pd.DataFrame.from_records(records)
    df = _ensure_time_key(df)
    return df


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--run_id", type=str, default="latest")
    ap.add_argument("--duration_s", type=float, default=60.0)
    ap.add_argument("--dt_s", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--n_vehicles", type=int, default=60)
    ap.add_argument("--speed_mps", type=float, default=15.0)
    ap.add_argument("--spacing_m", type=float, default=25.0)

    ap.add_argument("--refplus", action="store_true")
    ap.add_argument("--road_length_m", type=float, default=400.0)
    ap.add_argument("--n_lanes_per_dir", type=int, default=3)
    ap.add_argument("--lane_width_m", type=float, default=3.5)
    ap.add_argument("--median_gap_m", type=float, default=1.0)
    ap.add_argument("--s_curve_x0", type=float, default=120.0)
    ap.add_argument("--s_curve_x1", type=float, default=280.0)
    ap.add_argument("--s_curve_amp_y_m", type=float, default=20.0)
    ap.add_argument("--i1_x", type=float, default=160.0)
    ap.add_argument("--i2_x", type=float, default=240.0)
    ap.add_argument("--intersection_zone_m", type=float, default=80.0)

    ap.add_argument("--traffic_idm", action="store_true")
    ap.add_argument("--flow_main_vph", type=float, default=900.0)
    ap.add_argument("--veh_length_m", type=float, default=4.5)
    ap.add_argument("--min_spawn_gap_m", type=float, default=12.0)
    ap.add_argument("--enable_vehicle_mix", action="store_true")
    ap.add_argument("--car_ratio", type=float, default=0.85)
    ap.add_argument("--heavy_length_m", type=float, default=11.5)

    ap.add_argument("--idm_v0_mps", type=float, default=16.0)
    ap.add_argument("--idm_T_s", type=float, default=1.2)
    ap.add_argument("--idm_a_mps2", type=float, default=1.2)
    ap.add_argument("--idm_b_mps2", type=float, default=2.0)
    ap.add_argument("--idm_s0_m", type=float, default=2.0)
    ap.add_argument("--idm_delta", type=float, default=4.0)

    ap.add_argument("--traffic_signals", action="store_true")
    ap.add_argument("--sig_cycle_s", type=float, default=90.0)
    ap.add_argument("--sig_green_main_s", type=float, default=55.0)
    ap.add_argument("--sig_all_red_s", type=float, default=2.0)
    ap.add_argument("--sig_offset_i2_s", type=float, default=15.0)

    ap.add_argument("--cross_enable", action="store_true")
    ap.add_argument("--flow_cross_vph", type=float, default=700.0)
    ap.add_argument("--cross_half_length_m", type=float, default=350.0)
    ap.add_argument("--p_turn_i1", type=float, default=0.20)
    ap.add_argument("--p_turn_i2", type=float, default=0.20)
    ap.add_argument("--p_right", type=float, default=0.50)
    ap.add_argument("--p_left", type=float, default=0.50)

    args = ap.parse_args()

    run_id = _pick_run_id(args.run_id)
    rp = ensure_run_dirs_a(run_id, meta={"script": "generate_trajectories_C.py"})
    log_command(run_id, rp.run_results_dir)

    update_manifest(
        rp.manifest_path,
        {
            "trajectories": {
                "duration_s": float(args.duration_s),
                "dt_s": float(args.dt_s),
                "seed": int(args.seed),
                "refplus": bool(args.refplus),
                "traffic_idm": bool(args.traffic_idm),
                "traffic_signals": bool(args.traffic_signals),
                "cross_enable": bool(args.cross_enable),
                "params": vars(args),
            }
        },
    )

    if args.refplus:
        geom = RefPlusGeometry(
            road_length_m=float(args.road_length_m),
            n_lanes_per_dir=int(args.n_lanes_per_dir),
            lane_width_m=float(args.lane_width_m),
            median_gap_m=float(args.median_gap_m),
            s_curve_x0=float(args.s_curve_x0),
            s_curve_x1=float(args.s_curve_x1),
            s_curve_amp_y_m=float(args.s_curve_amp_y_m),
            i1_x=float(args.i1_x),
            i2_x=float(args.i2_x),
            intersection_zone_m=float(args.intersection_zone_m),
        )

        idm_params = IDMParams(
            v0_mps=float(args.idm_v0_mps),
            T_s=float(args.idm_T_s),
            a_mps2=float(args.idm_a_mps2),
            b_mps2=float(args.idm_b_mps2),
            s0_m=float(args.idm_s0_m),
            delta=float(args.idm_delta),
        )

        enable_signals = bool(args.traffic_signals)

        plan_i1 = SignalPlan(
            cycle_s=float(args.sig_cycle_s),
            green_main_s=float(args.sig_green_main_s),
            all_red_s=float(args.sig_all_red_s),
            offset_s=0.0,
        )
        plan_i2 = SignalPlan(
            cycle_s=float(args.sig_cycle_s),
            green_main_s=float(args.sig_green_main_s),
            all_red_s=float(args.sig_all_red_s),
            offset_s=float(args.sig_offset_i2_s),
        )

        df = _simulate_refplus_idm(
            geom=geom,
            plan_i1=plan_i1,
            plan_i2=plan_i2,
            flow_main_vph=float(args.flow_main_vph) if args.traffic_idm else 0.0,
            flow_cross_vph=float(args.flow_cross_vph) if args.cross_enable else 0.0,
            p_turn_i1=float(args.p_turn_i1),
            p_turn_i2=float(args.p_turn_i2),
            p_right=float(args.p_right),
            p_left=float(args.p_left),
            veh_length_m=float(args.veh_length_m),
            min_spawn_gap_m=float(args.min_spawn_gap_m),
            duration_s=float(args.duration_s),
            dt_s=float(args.dt_s),
            seed=int(args.seed),
            enable_signals=enable_signals,
            cross_enable=bool(args.cross_enable),
            cross_half_length_m=float(args.cross_half_length_m),
            idm_params=idm_params,
            enable_vehicle_mix=bool(args.enable_vehicle_mix),
            car_ratio=float(args.car_ratio),
            heavy_length_m=float(args.heavy_length_m),
        )

        out_path = rp.traj_dir / "traj__Ref.csv"
        df.to_csv(out_path, index=False)
        snapshot_file(out_path, rp.run_results_dir, category="trajectories")

        print(f"[OK] run_id={run_id}")
        print(f"[OK] traj -> {out_path} (rows={len(df)})")
        return

    # fallback: simple straight-line toy trajectories (kept for compatibility)
    times = np.arange(0.0, float(args.duration_s) + 1e-9, float(args.dt_s))
    records = []
    for vid in range(1, int(args.n_vehicles) + 1):
        x0 = (vid - 1) * float(args.spacing_m)
        for t in times:
            x = x0 + float(args.speed_mps) * float(t)
            records.append(
                {
                    "time_s": float(t),
                    "time_key": round(float(t), 3),
                    "veh_id": int(vid),
                    "x_m": float(x),
                    "y_m": 0.0,
                    "speed_mps": float(args.speed_mps),
                    "road_tag": "ToyMain",
                }
            )

    df = pd.DataFrame.from_records(records)
    out_path = rp.traj_dir / "traj__Ref.csv"
    df.to_csv(out_path, index=False)
    snapshot_file(out_path, rp.run_results_dir, category="trajectories")
    print(f"[OK] run_id={run_id}")
    print(f"[OK] traj -> {out_path} (rows={len(df)})")


if __name__ == "__main__":
    main()