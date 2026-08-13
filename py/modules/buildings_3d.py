# modules/buildings_3d.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rect2D:
    """Axis-aligned rectangle footprint in the x-y plane."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def normalized(self) -> "Rect2D":
        return Rect2D(
            x_min=min(self.x_min, self.x_max),
            x_max=max(self.x_min, self.x_max),
            y_min=min(self.y_min, self.y_max),
            y_max=max(self.y_min, self.y_max),
        )


@dataclass(frozen=True)
class BuildingBlock(Rect2D):
    """
    Minimal 3D building block: footprint + height + zone tag.

    Propagation currently only needs the 2D footprint (d_min),
    but height/zone are useful for logging/visualization/future extensions.
    """
    bid: int = 0
    height_m: float = 20.0
    zone: str = "Normal"

    def as_row(self) -> dict:
        r = self.normalized()
        return {
            "bid": int(self.bid),
            "x_min": float(r.x_min),
            "x_max": float(r.x_max),
            "y_min": float(r.y_min),
            "y_max": float(r.y_max),
            "height_m": float(self.height_m),
            "zone": str(self.zone),
        }


def _pick_zone_by_x(x_mid: float, road_length_m: float) -> str:
    """Simple deterministic zoning rule by longitudinal position."""
    L = max(1e-6, float(road_length_m))
    u = float(np.clip(x_mid / L, 0.0, 1.0))
    # 3 bands: Residential / CBD / Industrial
    if u < 0.30:
        return "Residential"
    if u < 0.70:
        return "CBD"
    return "Industrial"


def generate_buildings(
    *,
    seed: int,
    road_length_m: float,
    n_blocks: int = 12,
    x_margin_m: float = 10.0,
    y_halfspan_mode: str = "cross_road",   # "cross_road" or "one_side"
    min_w_m: float = 18.0,
    max_w_m: float = 45.0,
    min_h_m: float = 8.0,
    max_h_m: float = 22.0,
    min_height_m: float = 10.0,
    max_height_m: float = 50.0,
) -> list[BuildingBlock]:
    """
    Generate a set of rectangular building blocks.

    Spirit is aligned with your current generator script, but placed into a reusable module,
    with a better zone tagging rule.
    """
    rng = np.random.default_rng(int(seed))
    L = float(road_length_m)

    blocks: list[BuildingBlock] = []
    for bid in range(int(n_blocks)):
        w = float(rng.uniform(min_w_m, max_w_m))
        h = float(rng.uniform(min_h_m, max_h_m))

        x_min = float(rng.uniform(x_margin_m, max(x_margin_m, L - x_margin_m - w)))
        x_max = x_min + w

        if y_halfspan_mode == "cross_road":
            y_min, y_max = -h, +h
        elif y_halfspan_mode == "one_side":
            side = float(rng.choice([-1.0, +1.0]))
            y_min = side * 2.0
            y_max = side * (2.0 + h)
        else:
            raise ValueError("y_halfspan_mode must be 'cross_road' or 'one_side'")

        height_m = float(rng.uniform(min_height_m, max_height_m))
        zone = _pick_zone_by_x(0.5 * (x_min + x_max), L)

        blocks.append(
            BuildingBlock(
                bid=int(bid),
                x_min=float(x_min),
                x_max=float(x_max),
                y_min=float(y_min),
                y_max=float(y_max),
                height_m=float(height_m),
                zone=str(zone),
            )
        )
    return blocks


def buildings_to_dataframe(buildings: Iterable[BuildingBlock]) -> pd.DataFrame:
    rows = [b.as_row() for b in buildings]
    return pd.DataFrame(rows, columns=["bid", "x_min", "x_max", "y_min", "y_max", "height_m", "zone"])


def save_buildings_csv(buildings: Iterable[BuildingBlock], path: Path) -> Path:
    df = buildings_to_dataframe(buildings)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_buildings_csv(path: Path) -> list[BuildingBlock]:
    df = pd.read_csv(path)
    need = {"x_min", "x_max", "y_min", "y_max"}
    if not need.issubset(df.columns):
        raise ValueError(f"buildings missing columns {need}: {path}")

    out: list[BuildingBlock] = []
    for i, r in df.iterrows():
        bid = int(r["bid"]) if "bid" in df.columns else int(i)
        height_m = float(r["height_m"]) if "height_m" in df.columns else 20.0
        zone = str(r["zone"]) if "zone" in df.columns else "Normal"
        out.append(
            BuildingBlock(
                bid=bid,
                x_min=float(r["x_min"]),
                x_max=float(r["x_max"]),
                y_min=float(r["y_min"]),
                y_max=float(r["y_max"]),
                height_m=height_m,
                zone=zone,
            )
        )
    return out


def as_rects(buildings: Iterable[BuildingBlock]) -> list[Rect2D]:
    """Return only 2D footprints, for propagation computations."""
    return [Rect2D(b.x_min, b.x_max, b.y_min, b.y_max).normalized() for b in buildings]