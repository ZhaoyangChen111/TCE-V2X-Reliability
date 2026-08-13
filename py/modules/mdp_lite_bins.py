from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class MdpLiteBinningConfig:
    distance_edges_m: Tuple[float, ...] = (0.0, 50.0, 100.0, 150.0, 200.0)
    slack_edges_ms: Tuple[float, ...] = (-1.0e9, -50.0, 0.0, 20.0, 1.0e9)
    load_edges: Tuple[float, ...] = (0.0, 0.20, 0.40, 0.70, 1.000001)


def _clip_edges(edges: Iterable[float]) -> Tuple[float, ...]:
    arr = np.asarray(list(edges), dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("bin edges must be a 1-D iterable with at least two entries")
    if not np.all(np.diff(arr) > 0):
        raise ValueError("bin edges must be strictly increasing")
    return tuple(float(x) for x in arr.tolist())


def validate_binning_config(cfg: MdpLiteBinningConfig) -> MdpLiteBinningConfig:
    return MdpLiteBinningConfig(
        distance_edges_m=_clip_edges(cfg.distance_edges_m),
        slack_edges_ms=_clip_edges(cfg.slack_edges_ms),
        load_edges=_clip_edges(cfg.load_edges),
    )


def _fmt_edge(x: float) -> str:
    if abs(x - 1.0e9) < 1.0:
        return "INF"
    if abs(x + 1.0e9) < 1.0:
        return "MIN"
    s = str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:g}"
    return s.replace('-', 'm').replace('.', 'p')


def _band_label(value: float, edges: Tuple[float, ...], prefix: str) -> str:
    if not np.isfinite(value):
        return f"{prefix}NA"
    v = float(value)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if v <= hi or hi == edges[-1]:
            return f"{prefix}{_fmt_edge(lo)}_{_fmt_edge(hi)}"
    return f"{prefix}{_fmt_edge(edges[-2])}_{_fmt_edge(edges[-1])}"


def distance_band_label(distance_m: float, cfg: MdpLiteBinningConfig | None = None) -> str:
    cfg = validate_binning_config(cfg or MdpLiteBinningConfig())
    return _band_label(distance_m, cfg.distance_edges_m, "d")


def slack_band_label(slack_ms: float, cfg: MdpLiteBinningConfig | None = None) -> str:
    cfg = validate_binning_config(cfg or MdpLiteBinningConfig())
    return _band_label(slack_ms, cfg.slack_edges_ms, "s")


def load_band_label(cbr: float, cfg: MdpLiteBinningConfig | None = None) -> str:
    cfg = validate_binning_config(cfg or MdpLiteBinningConfig())
    return _band_label(cbr, cfg.load_edges, "c")


def make_state_key(
    *,
    remaining_budget: int,
    distance_m: float,
    slack_ms: float,
    cbr: float,
    cfg: MdpLiteBinningConfig | None = None,
) -> str:
    cfg = validate_binning_config(cfg or MdpLiteBinningConfig())
    r = max(int(remaining_budget), 0)
    return "__".join(
        [
            f"r{r}",
            distance_band_label(distance_m, cfg),
            slack_band_label(slack_ms, cfg),
            load_band_label(cbr, cfg),
        ]
    )


def decompose_state_key(key: str) -> dict:
    parts = str(key).split("__")
    out = {"remaining_budget": None, "distance_band": None, "slack_band": None, "load_band": None}
    for p in parts:
        if p.startswith("r") and p[1:].isdigit():
            out["remaining_budget"] = int(p[1:])
        elif p.startswith("d"):
            out["distance_band"] = p
        elif p.startswith("s"):
            out["slack_band"] = p
        elif p.startswith("c"):
            out["load_band"] = p
    return out



def state_signature(
    *,
    remaining_budget: int,
    distance_m: float,
    slack_ms: float,
    cbr: float,
    cfg: MdpLiteBinningConfig | None = None,
) -> dict:
    cfg = validate_binning_config(cfg or MdpLiteBinningConfig())
    return {
        'remaining_budget': max(int(remaining_budget), 0),
        'distance_band': distance_band_label(distance_m, cfg),
        'slack_band': slack_band_label(slack_ms, cfg),
        'load_band': load_band_label(cbr, cfg),
    }
