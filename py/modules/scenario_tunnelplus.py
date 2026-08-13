# modules/scenario_tunnelplus.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field

from .prop_tunnel import TunnelConfig


@dataclass(frozen=True)
class TunnelScenarioConfig:
    """Tunnel+ config (Gate-B: tunnel impairment + long-tail delay)."""
    tunnel: TunnelConfig = field(default_factory=TunnelConfig)

    def to_manifest(self) -> dict:
        d = asdict(self)
        d["name"] = "TunnelPlus"
        return d