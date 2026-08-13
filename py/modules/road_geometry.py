# modules/road_geometry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

Number = Union[float, int, np.ndarray]


@dataclass(frozen=True)
class RefPlusGeometry:
    """
    RefPlus geometry:
      - Main road: S-curve centerline, multi-lane (both directions)
      - Two intersections I1/I2 on main road
      - Cross roads: vertical roads crossing main at I1/I2
      - Turning: simple cubic Bezier paths (Right/Left) for visualization + congestion diversity
    """
    road_length_m: float = 3000.0
    n_lanes_per_dir: int = 3
    lane_width_m: float = 3.5
    median_gap_m: float = 2.0

    # S-curve on main
    s_curve_x0: float = 900.0
    s_curve_x1: float = 1500.0
    s_curve_amp_y_m: float = 12.0

    # intersections on main
    i1_x: float = 1000.0
    i2_x: float = 2000.0
    intersection_zone_m: float = 60.0

    # stopline offset (before intersection center, along the approach direction)
    stopline_offset_m: float = 20.0

    # Gate-C: cross roads
    cross_half_length_m: float = 400.0  # vertical road length = 2*half
    n_cross_lanes_per_dir: int = 1

    # turning shape
    turn_radius_m: float = 55.0  # controls turn curve size (roughly the "radius")

    # ---------------- Main road ----------------

    def centerline_y(self, x: Number) -> Number:
        x_arr = np.asarray(x, dtype=float)
        y = np.zeros_like(x_arr)
        x0 = float(self.s_curve_x0)
        x1 = float(self.s_curve_x1)
        if x1 > x0:
            m = (x_arr >= x0) & (x_arr <= x1)
            if np.any(m):
                u = (x_arr[m] - x0) / (x1 - x0)
                y[m] = float(self.s_curve_amp_y_m) * np.sin(2.0 * np.pi * u)
        if np.isscalar(x):
            return float(y.reshape(-1)[0])
        return y

    def lane_center_y(self, x: Number, direction: int, lane_id: int) -> Number:
        n = int(self.n_lanes_per_dir)
        if not (0 <= int(lane_id) < n):
            raise ValueError(f"lane_id out of range: {lane_id}")
        side = 1.0 if int(direction) >= 0 else -1.0
        offset = (float(lane_id) - (n - 1) / 2.0) * float(self.lane_width_m)
        base = self.centerline_y(x)
        return base + side * (float(self.median_gap_m) / 2.0 + offset)

    def road_tag(self, x: float) -> str:
        if abs(float(x) - float(self.i1_x)) <= float(self.intersection_zone_m):
            return "I1_ZONE"
        if abs(float(x) - float(self.i2_x)) <= float(self.intersection_zone_m):
            return "I2_ZONE"
        if float(self.s_curve_x0) <= float(x) <= float(self.s_curve_x1):
            return "S_CURVE"
        return "MAIN"

    def stopline_x(self, which: str, direction: int) -> float:
        cx = float(self.i1_x) if which.upper() == "I1" else float(self.i2_x)
        d = float(self.stopline_offset_m)
        # direction +1 approaches from left to right, stopline is before center: cx - d
        # direction -1 approaches from right to left, stopline is before center: cx + d
        return cx - d if int(direction) >= 0 else cx + d

    # ---------------- Cross roads (vertical) ----------------

    def cross_center_x(self, which: str) -> float:
        return float(self.i1_x) if which.upper() == "I1" else float(self.i2_x)

    def cross_lane_center_x(self, which: str, direction: int, lane_id: int = 0) -> float:
        n = int(self.n_cross_lanes_per_dir)
        if not (0 <= int(lane_id) < n):
            raise ValueError(f"cross lane_id out of range: {lane_id}")
        side = 1.0 if int(direction) >= 0 else -1.0
        offset = (float(lane_id) - (n - 1) / 2.0) * float(self.lane_width_m)
        cx = self.cross_center_x(which)
        return cx + side * (float(self.median_gap_m) / 2.0 + offset)

    def cross_xy(self, s: Number, which: str, direction: int, lane_id: int = 0) -> tuple[Number, Number]:
        """
        Cross road along y-axis.
        We use s in [0, 2*half], where intersection center y=0 corresponds to s=half.
          direction +1: from y=-half -> +half
          direction -1: from y=+half -> -half
        """
        s_arr = np.asarray(s, dtype=float)
        half = float(self.cross_half_length_m)
        if int(direction) >= 0:
            y = -half + s_arr
        else:
            y = +half - s_arr
        x = np.full_like(y, self.cross_lane_center_x(which, direction, lane_id), dtype=float)
        if np.isscalar(s):
            return float(x.reshape(-1)[0]), float(y.reshape(-1)[0])
        return x, y

    def cross_stopline_s(self) -> float:
        # stopline is before y=0 by stopline_offset along approach direction
        half = float(self.cross_half_length_m)
        return max(0.0, half - float(self.stopline_offset_m))

    # ---------------- Turning (Bezier) ----------------

    @staticmethod
    def bezier_xy(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: float) -> np.ndarray:
        u = float(np.clip(u, 0.0, 1.0))
        a = (1.0 - u)
        return (a**3) * p0 + 3.0 * (a**2) * u * p1 + 3.0 * a * (u**2) * p2 + (u**3) * p3

    def _which_to_x(self, which: int | str) -> float:
        if isinstance(which, str):
            w = which.upper().replace("I", "")
            which_i = int(w)
        else:
            which_i = int(which)
        return float(self.i1_x) if which_i == 1 else float(self.i2_x)

    def _turn_ctrl_points(self, which: int | str, turn_kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a simple cubic Bezier curve for turning.
        turn_kind: 'R' (to +x direction) or 'L' (to -x direction).
        """
        cx = self._which_to_x(which)
        tk = str(turn_kind).upper()
        R = float(self.turn_radius_m)
        kappa = 0.5522847498  # quarter circle bezier constant

        # start point at cross road center (x=cx, y=0)
        p0 = np.array([cx, 0.0], dtype=float)

        # end point goes to main road lane-0 center a bit away from intersection
        if tk == "R":
            x_end = cx + R
            direction = +1
            p3 = np.array([x_end, float(self.lane_center_y(x_end, direction, 0))], dtype=float)
            p1 = p0 + np.array([0.0, +R * kappa], dtype=float)
            p2 = p3 + np.array([-R * kappa, 0.0], dtype=float)
        else:
            # treat everything else as left
            x_end = cx - R
            direction = -1
            p3 = np.array([x_end, float(self.lane_center_y(x_end, direction, 0))], dtype=float)
            p1 = p0 + np.array([0.0, +R * kappa], dtype=float)
            p2 = p3 + np.array([+R * kappa, 0.0], dtype=float)

        return p0, p1, p2, p3

    def turn_xy(self, which: int | str, turn_kind: str, u: float) -> tuple[float, float]:
        """
        Return (x,y) on the turning curve at parameter u in [0,1].
        """
        p0, p1, p2, p3 = self._turn_ctrl_points(which, turn_kind)
        p = self.bezier_xy(p0, p1, p2, p3, float(u))
        return float(p[0]), float(p[1])

    def turn_path_length_m(self, which: int | str, turn_kind: str, n_samples: int = 80) -> float:
        """
        Approximate arc length of the turning curve by sampling.
        Deterministic and stable (used by generate_trajectories_A.py).
        """
        n = max(10, int(n_samples))
        us = np.linspace(0.0, 1.0, n)
        pts = np.array([self.turn_xy(which, turn_kind, float(u)) for u in us], dtype=float)
        d = np.diff(pts, axis=0)
        seg = np.hypot(d[:, 0], d[:, 1])
        return float(np.sum(seg))

    # helper: main road intersection "s" coordinate (used by some older scripts)
    def main_intersection_s(self, which: str, direction: int) -> float:
        cx = float(self.i1_x) if which.upper() == "I1" else float(self.i2_x)
        L = float(self.road_length_m)
        return cx if int(direction) >= 0 else (L - cx)