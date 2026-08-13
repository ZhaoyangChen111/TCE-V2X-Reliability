# Methodology

## TCE

`py/modules/tce_metric.py` is the canonical offline definition. Physical
reception is read from `success_phy`; timeliness is recomputed using finite delay
and the active deadline. A late packet inside the grace interval retains
`exp[-beta((d-D)/G)^gamma]`; a lost or later packet has zero utility. At `d=D+G`
the utility is `exp(-beta)` and becomes zero only for `d>D+G`.

The S15 configuration is D20/G50, beta=ln(20), gamma=2. The retransmission
controller uses the same-shaped predicted delay utility, while the final TCE
analysis is offline.

## UDRC and C_i

UDRC estimates remaining-chain gain and cost and retransmits iff

```text
sum(reach_a * p_a * U_a)
  - lambda * sum(reach_a * C_a * 0.65^(future step)) > 0.
```

For horizon `H=max(D+G,D)`, `C_i` starts with normalized incremental delay
`delta_delay/H`. Depending on cost mode it adds normalized airtime and scales
these terms with a quadratic busy-pressure proxy, optionally including collision
probability. The decision log exposes every cost component. The S15 working point
uses lambda=0.03 and `delay_cbr`.

## Policy semantics

- NoRet: no continuation.
- Classical: continue whenever retry budget remains.
- Nomikos: conceptual hard-deadline baseline; continue only if predicted next delay is timely.
- UDRC: chain utility-minus-cost decision.
- MDP-lite: empirical transition lookup plus short recursive value; strict release mode does not silently fall back.

## Scope

Propagation, congestion, traffic, and retransmission are system-level
abstractions. No protocol stack or application-history utility is implemented.
