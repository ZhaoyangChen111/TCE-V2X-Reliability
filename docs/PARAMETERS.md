# Parameters

## Formal S15 values

The machine-readable source is `configs/paper_s15.json`. Key values are 15 seeds,
60 s, 10 Hz, 200 m, D20/G50, beta=ln(20), gamma=2, mixed TX selection, UDRC
lambda=0.03, and MDP-lite cost scale=0.85.

## Parameter classes

Routine run controls include scenario selection, congestion switch, run ID,
duration, seed range, TX count, retry budget, and plotting switches.

Scientific configuration includes TCE D/G/beta/gamma; propagation, traffic and
congestion parameters; UDRC lambda and cost mode; MDP model, threshold, cost scale,
minimum samples and discount. Changing them is supported, but the result must not
be labelled an S15 reproduction without a corresponding new experiment record.

Use `python py/run_pipeline_C.py --help` for all options. Prefer the reviewed
examples instead of relying on generic defaults.
