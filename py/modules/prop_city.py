from __future__ import annotations

from math import erf, sqrt, pi
from typing import Iterable, Protocol, Tuple
import numpy as np


class RectLike(Protocol):
    x_min: float
    x_max: float
    y_min: float
    y_max: float


# -----------------------------------------------------------------------------
# Urban propagation model used in 03_sim_B3
# -----------------------------------------------------------------------------
# Same-street urban links:
#   - 3GPP TR 37.885 urban LOS / NLOSv state logic and path-loss formulas.
# Cross-street / intersection NLOS links:
#   - VirtualSource11p-inspired NLOS model (Mangel et al., 2011), with
#     validation context from Abbas et al. (2013).
# Building-blocked same-street links:
#   - 3GPP urban NLOS path-loss, optionally softened by a reflection-recovery
#     term to preserve a low-complexity geometry-aware surrogate.
# Path-loss -> packet success:
#   - log-normal outage mapping using state-dependent shadowing sigma.
# -----------------------------------------------------------------------------


def clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


# -----------------------------
# Geometry helpers
# -----------------------------

def segment_intersects_rect(ax, ay, bx, by, rect: RectLike) -> bool:
    ax, ay, bx, by = map(float, [ax, ay, bx, by])
    rx0, rx1 = float(rect.x_min), float(rect.x_max)
    ry0, ry1 = float(rect.y_min), float(rect.y_max)

    seg_xmin, seg_xmax = (ax, bx) if ax <= bx else (bx, ax)
    seg_ymin, seg_ymax = (ay, by) if ay <= by else (by, ay)
    if seg_xmax < rx0 or seg_xmin > rx1 or seg_ymax < ry0 or seg_ymin > ry1:
        return False

    if (rx0 <= ax <= rx1 and ry0 <= ay <= ry1) or (rx0 <= bx <= rx1 and ry0 <= by <= ry1):
        return True

    def ccw(x1, y1, x2, y2, x3, y3):
        return (y3 - y1) * (x2 - x1) > (y2 - y1) * (x3 - x1)

    def intersect(x1, y1, x2, y2, x3, y3, x4, y4):
        return (
            ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4)
            and ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4)
        )

    edges = [
        (rx0, ry0, rx1, ry0),
        (rx1, ry0, rx1, ry1),
        (rx1, ry1, rx0, ry1),
        (rx0, ry1, rx0, ry0),
    ]
    for (x3, y3, x4, y4) in edges:
        if intersect(ax, ay, bx, by, x3, y3, x4, y4):
            return True
    return False


def point_to_rect_distance(px: float, py: float, rect: RectLike) -> float:
    px, py = float(px), float(py)
    rx0, rx1 = float(rect.x_min), float(rect.x_max)
    ry0, ry1 = float(rect.y_min), float(rect.y_max)
    dx = 0.0
    if px < rx0:
        dx = rx0 - px
    elif px > rx1:
        dx = px - rx1
    dy = 0.0
    if py < ry0:
        dy = ry0 - py
    elif py > ry1:
        dy = py - ry1
    return float(np.hypot(dx, dy))


def point_to_buildings_min_distance(px: float, py: float, buildings: Iterable[RectLike]) -> float:
    blds = list(buildings) if buildings is not None else []
    if not blds:
        return float("inf")
    d = float("inf")
    for rect in blds:
        d = min(d, point_to_rect_distance(px, py, rect))
    return float(d)


def segment_to_rect_min_distance(ax, ay, bx, by, rect: RectLike) -> float:
    ax, ay, bx, by = map(float, [ax, ay, bx, by])
    rx0, rx1 = float(rect.x_min), float(rect.x_max)
    ry0, ry1 = float(rect.y_min), float(rect.y_max)

    if segment_intersects_rect(ax, ay, bx, by, rect):
        return 0.0

    def point_segment_dist(px, py, x1, y1, x2, y2) -> float:
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return float(np.hypot(px - x1, py - y1))
        c2 = vx * vx + vy * vy
        if c2 <= c1:
            return float(np.hypot(px - x2, py - y2))
        t = c1 / c2
        projx, projy = x1 + t * vx, y1 + t * vy
        return float(np.hypot(px - projx, py - projy))

    corners = [(rx0, ry0), (rx0, ry1), (rx1, ry0), (rx1, ry1)]
    d = min(point_to_rect_distance(ax, ay, rect), point_to_rect_distance(bx, by, rect))
    for (cx, cy) in corners:
        d = min(d, point_segment_dist(cx, cy, ax, ay, bx, by))
    return float(d)


def blockage_strength_with_dmin(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    buildings: Iterable[RectLike],
    transition_m: float,
) -> Tuple[float, float]:
    blds = list(buildings) if buildings is not None else []
    if not blds:
        return 0.0, float("inf")

    d_min = float("inf")
    intersects = False
    for rect in blds:
        d = segment_to_rect_min_distance(ax, ay, bx, by, rect)
        if d < d_min:
            d_min = d
        if d == 0.0:
            intersects = True
            break

    if intersects:
        return 1.0, 0.0
    t = max(float(transition_m), 1e-6)
    return clamp01(np.exp(-((d_min / t) ** 2))), float(d_min)


# -----------------------------
# 3GPP-inspired urban V2V model
# -----------------------------

def road_family(tag: str) -> str:
    t = (tag or "").strip().upper()
    if not t:
        return ""
    if t.startswith("MAIN_"):
        return "MAIN"
    if t.startswith("CROSS_I1_"):
        return "CROSS_I1"
    if t.startswith("CROSS_I2_"):
        return "CROSS_I2"
    if t.startswith("TURN_I1_"):
        return "TURN_I1"
    if t.startswith("TURN_I2_"):
        return "TURN_I2"
    return t


def is_same_street(tx_road_tag: str, rx_road_tag: str) -> bool:
    a = road_family(tx_road_tag)
    b = road_family(rx_road_tag)
    return bool(a) and bool(b) and a == b


def urban_los_probability_3gpp(distance_m: float) -> float:
    d = max(float(distance_m), 0.0)
    return clamp01(min(1.0, 1.05 * np.exp(-0.0114 * d)))


# V2V urban LOS/NLOS path-loss from 3GPP TR 37.885.
# For NLOSv the LOS path-loss is reused, plus an additional vehicle-blockage loss.
def urban_pathloss_los_3gpp(distance_m: float, fc_ghz: float = 5.9) -> float:
    d3d = max(float(distance_m), 1.0)
    fc = max(float(fc_ghz), 0.1)
    return float(38.77 + 16.7 * np.log10(d3d) + 18.2 * np.log10(fc))


def urban_pathloss_nlos_3gpp(distance_m: float, fc_ghz: float = 5.9) -> float:
    d3d = max(float(distance_m), 1.0)
    fc = max(float(fc_ghz), 0.1)
    return float(36.85 + 30.0 * np.log10(d3d) + 18.9 * np.log10(fc))


def urban_nlosv_blockage_stats_3gpp(distance_m: float, case: str = "case3") -> Tuple[float, float]:
    d = max(float(distance_m), 1.0)
    extra = max(0.0, 15.0 * np.log10(d) - 41.0)
    c = str(case).strip().lower()
    if c in ("case2", "high_blocker", "lower"):
        return float(9.0 + extra), 4.5
    if c in ("case1", "no_blockage"):
        return 0.0, 0.0
    return float(5.0 + extra), 4.0  # Case 3: default when blocker-height details are unavailable


def sample_nlosv_blockage_loss_db(
    rng: np.random.Generator,
    distance_m: float,
    case: str = "case3",
) -> float:
    mu_db, sigma_db = urban_nlosv_blockage_stats_3gpp(distance_m, case=case)
    if sigma_db <= 1e-9:
        return max(0.0, float(mu_db))
    x_db = float(rng.normal(loc=float(mu_db), scale=float(sigma_db)))
    return max(0.0, x_db)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(float(x) / sqrt(2.0)))


def success_probability_from_outage(
    mean_pathloss_db: float,
    threshold_db: float,
    sigma_sf_db: float,
    min_prob: float = 0.001,
    max_prob: float = 0.999,
) -> float:
    sigma = max(float(sigma_sf_db), 1e-6)
    z = (float(threshold_db) - float(mean_pathloss_db)) / sigma
    return float(np.clip(normal_cdf(z), float(min_prob), float(max_prob)))


def reflection_recovery_db(d_min_m: float, gmax_db: float, d0_m: float) -> float:
    if (not np.isfinite(d_min_m)) or float(gmax_db) <= 1e-9 or float(d0_m) <= 1e-9:
        return 0.0
    return float(max(0.0, float(gmax_db) * np.exp(-float(d_min_m) / float(d0_m))))


# -----------------------------
# VirtualSource11p-inspired cross-street NLOS model
# -----------------------------
# Parameters from Mangel et al. (2011), with validation discussion in Abbas et al. (2013).
# We keep GID = 0 dB by default because Abbas et al. note that additional intersection-
# dependent calibration would require more site-specific measurement data.


def virtualsource11p_params() -> dict:
    return {
        "curve_shift_db": 3.75,
        "suburban_loss_db": 2.94,
        "loss_exponent": 2.69,
        "street_exponent": 0.81,
        "tx_distance_exponent": 0.957,
        "sigma_nlos_db": 4.1,
        "break_even_m": 180.0,
        "gid_db": 0.0,
    }


DEFAULT_INTERSECTION_CENTERS_X = (1000.0, 2000.0)
DEFAULT_MAIN_STREET_WIDTH_M = 23.0
DEFAULT_CROSS_STREET_WIDTH_M = 9.0
DEFAULT_TURN_STREET_WIDTH_M = 9.0
DEFAULT_NLOS_LOS_BLEND_M = 10.0


def _extract_intersection_from_tag(tag: str) -> int | None:
    t = (tag or "").upper()
    if "_I1_" in t:
        return 1
    if "_I2_" in t:
        return 2
    return None


def _receiver_street_width_m(
    rx_road_tag: str,
    main_street_width_m: float,
    cross_street_width_m: float,
    turn_street_width_m: float,
) -> float:
    fam = road_family(rx_road_tag)
    if fam == "MAIN":
        return float(main_street_width_m)
    if fam.startswith("CROSS_"):
        return float(cross_street_width_m)
    if fam.startswith("TURN_"):
        return float(turn_street_width_m)
    return float(main_street_width_m)


def _infer_intersection_center_x(
    tx_x: float,
    rx_x: float,
    tx_tag: str,
    rx_tag: str,
    centers_x: Tuple[float, float] = DEFAULT_INTERSECTION_CENTERS_X,
) -> float:
    itx = _extract_intersection_from_tag(tx_tag)
    irx = _extract_intersection_from_tag(rx_tag)
    centers = np.asarray(centers_x, dtype=float)
    if centers.size < 2:
        centers = np.asarray(DEFAULT_INTERSECTION_CENTERS_X, dtype=float)
    if itx is not None and irx is not None and itx == irx and 1 <= itx <= len(centers):
        return float(centers[itx - 1])
    if itx is not None and 1 <= itx <= len(centers):
        return float(centers[itx - 1])
    if irx is not None and 1 <= irx <= len(centers):
        return float(centers[irx - 1])
    mid_x = 0.5 * (float(tx_x) + float(rx_x))
    return float(centers[np.argmin(np.abs(centers - mid_x))])


def _distance_to_intersection_center(x: float, y: float, road_tag: str, center_x: float) -> float:
    fam = road_family(road_tag)
    if fam == "MAIN":
        return abs(float(x) - float(center_x))
    if fam.startswith("CROSS_") or fam.startswith("TURN_"):
        return abs(float(y))
    return float(np.hypot(float(x) - float(center_x), float(y)))


def virtualsource11p_pathloss_db(
    d_t_m: float,
    d_r_m: float,
    w_r_m: float,
    x_t_m: float,
    fc_ghz: float = 5.9,
    is_suburban: bool = False,
    gid_db: float = 0.0,
) -> float:
    p = virtualsource11p_params()
    d_t = max(float(d_t_m), 1.0)
    d_r = max(float(d_r_m), 1.0)
    w_r = max(float(w_r_m), 1.0)
    x_t = max(float(x_t_m), 1.0)
    fc_hz = max(float(fc_ghz), 0.1) * 1e9
    lambda_m = 3e8 / fc_hz
    inner = (d_t ** p["tx_distance_exponent"]) * (4.0 * pi * d_r) / ((x_t * (w_r ** p["street_exponent"])) * lambda_m)
    if d_r > p["break_even_m"]:
        inner = (d_t ** p["tx_distance_exponent"]) * (4.0 * pi * (d_r ** 2)) / (
            (x_t * (w_r ** p["street_exponent"])) * lambda_m * p["break_even_m"]
        )
    inner = max(inner, 1e-6)
    return float(
        p["curve_shift_db"]
        + (p["suburban_loss_db"] if bool(is_suburban) else 0.0)
        + float(gid_db)
        + 10.0 * p["loss_exponent"] * np.log10(inner)
    )


def classify_urbmask_link(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    buildings: Iterable[RectLike],
    transition_m: float,
    rng: np.random.Generator,
    tx_road_tag: str = "",
    rx_road_tag: str = "",
    fc_ghz: float = 5.9,
    pl50_los_db: float = 82.5,
    pl50_nlos_db: float = 100.0,
    enable_refl_gain: bool = False,
    gmax_db: float = 0.0,
    d0_m: float = 15.0,
    intersection_centers_x: Tuple[float, float] = DEFAULT_INTERSECTION_CENTERS_X,
    main_street_width_m: float = DEFAULT_MAIN_STREET_WIDTH_M,
    cross_street_width_m: float = DEFAULT_CROSS_STREET_WIDTH_M,
    turn_street_width_m: float = DEFAULT_TURN_STREET_WIDTH_M,
    nlos_los_blend_m: float = DEFAULT_NLOS_LOS_BLEND_M,
) -> dict:
    dist = float(np.hypot(float(bx) - float(ax), float(by) - float(ay)))
    b_soft, d_min = blockage_strength_with_dmin(ax, ay, bx, by, buildings, transition_m)
    same_street = is_same_street(tx_road_tag, rx_road_tag) if (tx_road_tag or rx_road_tag) else True
    building_blocked = bool(np.isfinite(d_min) and d_min <= 1e-9)

    pl_los = urban_pathloss_los_3gpp(dist, fc_ghz=fc_ghz)
    pl_nlos = urban_pathloss_nlos_3gpp(dist, fc_ghz=fc_ghz)

    # 1) Cross-street / intersection links:
    #    use VirtualSource11p-inspired NLOS path loss, with a short transition to LOS
    #    near the intersection center as recommended in the original paper.
    if not same_street:
        center_x = _infer_intersection_center_x(ax, bx, tx_road_tag, rx_road_tag, centers_x=intersection_centers_x)
        d_t = _distance_to_intersection_center(ax, ay, tx_road_tag, center_x)
        d_r = _distance_to_intersection_center(bx, by, rx_road_tag, center_x)
        w_r = _receiver_street_width_m(
            rx_road_tag,
            main_street_width_m=main_street_width_m,
            cross_street_width_m=cross_street_width_m,
            turn_street_width_m=turn_street_width_m,
        )
        x_t = point_to_buildings_min_distance(ax, ay, buildings)
        if not np.isfinite(x_t):
            x_t = max(0.5 * float(transition_m), 3.0)
        x_t = max(float(x_t), 1.0)

        pl_vs = virtualsource11p_pathloss_db(
            d_t_m=d_t,
            d_r_m=d_r,
            w_r_m=w_r,
            x_t_m=x_t,
            fc_ghz=fc_ghz,
            is_suburban=False,
            gid_db=0.0,
        )
        # Close to the center the original paper recommends using LOS or a blend.
        # We adopt a short, deterministic LOS->NLOS transition window.
        los_mix = clamp01(min(d_t, d_r) / max(float(nlos_los_blend_m), 1e-6))
        mean_pl = (1.0 - los_mix) * pl_los + los_mix * pl_vs
        sigma_sf = (1.0 - los_mix) * 3.0 + los_mix * virtualsource11p_params()["sigma_nlos_db"]
        threshold_db = (1.0 - los_mix) * float(pl50_los_db) + los_mix * float(pl50_nlos_db)
        p_succ = success_probability_from_outage(
            mean_pathloss_db=mean_pl,
            threshold_db=threshold_db,
            sigma_sf_db=sigma_sf,
        )
        return {
            "distance_m": dist,
            "d_min_m": float(d_min),
            "blockage_b": float(los_mix),
            "link_state": "NLOS" if los_mix >= 0.5 else "LOS",
            "pathloss_db": float(mean_pl),
            "p_succ": p_succ,
            "p_los_geom": float(1.0 - los_mix),
            "p_los_3gpp": 0.0,
            "g_refl_db": 0.0,
            "same_street": same_street,
        }

    # 2) Same-street but building-blocked links:
    #    use 3GPP urban NLOS, optionally softened by an equivalent reflection aid.
    if building_blocked:
        g_refl_db = 0.0
        if enable_refl_gain:
            g_refl_db = reflection_recovery_db(d_min, gmax_db=gmax_db, d0_m=d0_m)
        mean_pl = max(pl_los, pl_nlos - g_refl_db)
        p_succ = success_probability_from_outage(
            mean_pathloss_db=mean_pl,
            threshold_db=float(pl50_nlos_db),
            sigma_sf_db=4.0,
        )
        return {
            "distance_m": dist,
            "d_min_m": float(d_min),
            "blockage_b": 1.0,
            "link_state": "NLOS",
            "pathloss_db": float(mean_pl),
            "p_succ": p_succ,
            "p_los_geom": 0.0,
            "p_los_3gpp": 0.0,
            "g_refl_db": float(g_refl_db),
            "same_street": same_street,
        }

    # 3) Same-street unobstructed links: 3GPP LOS/NLOSv.
    p_los = urban_los_probability_3gpp(dist)
    if rng.random() < p_los:
        mean_pl = pl_los
        sigma_sf = 3.0
        threshold_db = float(pl50_los_db)
        link_state = "LOS"
    else:
        veh_loss_db = sample_nlosv_blockage_loss_db(rng, dist, case="case3")
        mean_pl = pl_los + veh_loss_db
        sigma_sf = 3.0  # 3GPP reuses LOS shadowing model for NLOSv
        threshold_db = float(pl50_los_db)
        link_state = "NLOSv"

    p_succ = success_probability_from_outage(
        mean_pathloss_db=mean_pl,
        threshold_db=threshold_db,
        sigma_sf_db=sigma_sf,
    )
    return {
        "distance_m": dist,
        "d_min_m": float(d_min),
        "blockage_b": float(1.0 - p_los),
        "link_state": link_state,
        "pathloss_db": float(mean_pl),
        "p_succ": p_succ,
        "p_los_geom": 1.0,
        "p_los_3gpp": float(p_los),
        "g_refl_db": 0.0,
        "same_street": True,
    }
