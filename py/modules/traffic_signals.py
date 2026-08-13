# modules/traffic_signals.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalPlan:
    """
    Fixed-time signal plan (lightweight, deterministic).

    Phase order per cycle:
      1) MAIN_GREEN  (duration = green_main_s)
      2) ALL_RED     (duration = all_red_s)
      3) CROSS_GREEN (the remaining time in the cycle)

    Notes
    -----
    - We keep this module intentionally simple: no amber, no actuated control.
    - Key fix vs earlier version:
      ALL_RED must NOT be treated as CROSS_GREEN.
    """
    cycle_s: float = 90.0
    green_main_s: float = 55.0
    all_red_s: float = 2.0
    offset_s: float = 0.0

    def _tt(self, t: float) -> float:
        c = float(self.cycle_s)
        if c <= 0:
            return 0.0
        return (float(t) + float(self.offset_s)) % c

    def phase(self, t: float) -> str:
        tt = self._tt(t)
        g = float(self.green_main_s)
        ar = max(0.0, float(self.all_red_s))
        if tt < g:
            return "MAIN_GREEN"
        if tt < g + ar:
            return "ALL_RED"
        return "CROSS_GREEN"

    def main_is_green(self, t: float) -> bool:
        return self.phase(t) == "MAIN_GREEN"

    def main_is_red(self, t: float) -> bool:
        return not self.main_is_green(t)

    def cross_is_green(self, t: float) -> bool:
        return self.phase(t) == "CROSS_GREEN"

    def cross_is_red(self, t: float) -> bool:
        return not self.cross_is_green(t)