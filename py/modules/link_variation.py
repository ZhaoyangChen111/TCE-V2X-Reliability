from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def _seed_from_parts(*parts: object) -> int:
    s = "|".join(str(p) for p in parts)
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


@dataclass
class LinkVariationField:
    block_s: float = 0.5
    rho: float = 0.92
    sigma: float = 0.06
    clip_abs: float = 0.20
    base_seed: int = 1

    def __post_init__(self) -> None:
        self.block_s = max(0.05, float(self.block_s))
        self.rho = float(np.clip(float(self.rho), 0.0, 0.999))
        self.sigma = max(1e-6, float(self.sigma))
        self.clip_abs = max(0.01, float(self.clip_abs))
        self._state: dict[tuple[int, int], tuple[int, float]] = {}

    def _innovation(self, a: int, b: int, block_idx: int) -> float:
        seed = _seed_from_parts(self.base_seed, min(a, b), max(a, b), block_idx)
        rng = np.random.default_rng(seed)
        return float(rng.normal(0.0, self.sigma))

    def _step(self, prev: float, innov: float) -> float:
        var_scale = np.sqrt(max(1e-9, 1.0 - self.rho * self.rho))
        val = self.rho * float(prev) + var_scale * float(innov)
        return float(np.clip(val, -self.clip_abs, self.clip_abs))

    def get_bias(self, tx_id: int, rx_id: int, t_s: float, link_state: str = "LOS") -> float:
        pair = (min(int(tx_id), int(rx_id)), max(int(tx_id), int(rx_id)))
        block_idx = int(np.floor(float(t_s) / self.block_s + 1e-9))
        last = self._state.get(pair, None)

        if last is None:
            val = self._innovation(pair[0], pair[1], block_idx)
        else:
            last_block, val = last
            if block_idx > last_block:
                for kk in range(last_block + 1, block_idx + 1):
                    innov = self._innovation(pair[0], pair[1], kk)
                    val = self._step(val, innov)
            elif block_idx < last_block:
                val = self._innovation(pair[0], pair[1], block_idx)

        self._state[pair] = (block_idx, float(val))

        state = (link_state or "LOS").upper()
        if state == "LOS":
            mult = 0.8
        elif state == "NLOS":
            mult = 1.0
        else:
            mult = 1.1
        return float(np.clip(mult * float(val), -self.clip_abs, self.clip_abs))