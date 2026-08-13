# TCE-Driven Control for Timely V2X Safety Communications

English | [Français](README_FR.md)

## 1. Project overview

This repository contains the release candidate of a system-level
analytical/stochastic simulator for studying V2X safety-message reliability and
latency in degraded propagation and congestion conditions. It evaluates three
scenarios (`Ref`, `UrbMask`, and `Tunnel`), with optional congestion (`NoCong`
or `Cong`), and five retransmission policies.

The central metric is Timely Communication Effectiveness (TCE), which separates
strictly timely reception from reception that is late but still partially useful.
The code accompanies the research work *TCE-Driven Control for Timely V2X
Safety Communications*. No DOI is asserted here.

This is not a protocol-level V2X stack. See [Model scope and limitations](#13-model-scope-and-limitations).

## 2. What is TCE?

For a physically received packet with delay `d`, deadline `D`, grace interval
`G`, decay coefficient `beta`, and exponent `gamma`, the packet utility is

```text
u(d) = 1                                      if d <= D
       exp[-beta ((d-D)/G)^gamma]             if D < d <= D+G
       0                                      if d > D+G
```

A physically lost packet has utility zero. TCE is the mean packet utility.
The implementation recomputes timeliness from physical reception and the active
analysis deadline, and therefore satisfies, up to floating-point tolerance:

```text
Timely reception rate <= TCE <= PHY reception rate
```

PDR alone treats all physically received packets alike. TCE retains the
distinction between strictly timely, late-but-still-useful, and useless/lost
reception. The exact boundaries are documented in [Methodology](docs/METHODOLOGY.md).

## 3. Repository structure

```text
.
├── README.md
├── requirements.txt
├── configs/
│   └── paper_s15.json
├── docs/
├── examples/
│   ├── run_reviewer_smoke.py
│   └── run_paper_configuration.py
├── models/
│   ├── MODEL_MANIFEST.csv
│   └── six S15 empirical MDP-lite JSON models
├── matlab/
│   └── plot_s15_paper_figures.m
└── py/
    ├── run_pipeline_C.py
    ├── sim_v2x_C.py
    ├── analyze_metrics_C.py
    ├── analyze_policy_C.py
    ├── analyze_tce_C.py
    ├── compare_policies_C.py
    ├── study_readiness_C.py
    ├── build_mdp_lite_table_C.py
    └── modules/
        ├── retx_policy.py
        ├── tce_metric.py
        └── propagation, congestion, traffic, and scenario modules
```

Important entry points:

| File | Responsibility |
|---|---|
| `run_pipeline_C.py` | Unified generation → simulation → analysis → readiness → plotting pipeline. |
| `sim_v2x_C.py` | Packet-level stochastic simulation and decision-log generation. |
| `modules/retx_policy.py` | NoRet, Classical, Nomikos, UDRC, and MDP-lite decisions. |
| `modules/tce_metric.py` | Canonical TCE configuration, packet utility, and aggregation. |
| `analyze_metrics_C.py` | Conventional packet metrics and distance bands. |
| `analyze_policy_C.py` | Per-seed policy/TCE summaries and decision diagnostics. |
| `analyze_tce_C.py` | TCE summary, distance/band tables, and compact packet utilities. |
| `compare_policies_C.py` | Collects policy summaries into comparison tables. |
| `study_readiness_C.py` | Checks whether a D/G run exposes a useful hidden zone; this is a research-suitability check, not a software correctness test. |
| `build_mdp_lite_table_C.py` | Builds an empirical MDP-lite transition table offline. |
| `matlab/plot_s15_paper_figures.m` | Final S15 + MDP seven-figure plotting script. |

Generated files remain under `workspace/`, which is excluded from Git.

## 4. Installation

The release candidate was validated with Python 3.11.9 and the exact package
versions in `requirements.txt`.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Do not commit `.venv`. MATLAB is optional and used only for the retained paper
figure script. The MATLAB version was not formally pinned in the original experiments.

## 5. Quick start

Run the validated basic reviewer smoke from the repository root:

```powershell
python examples/run_reviewer_smoke.py
```

It invokes the real pipeline for a short `Ref`/Classical experiment using one
seed, a two-second trajectory, and safe `tx_mode=mix`. It then checks:

- packet rows are non-empty;
- the required packet schema exists;
- policy and TCE summaries exist;
- `Timely <= TCE <= PHY`;
- the run manifest retained pipeline, TCE, stage, and input provenance.

Success ends with `REVIEWER SMOKE PASS` and a run ID. Failure returns a non-zero
exit code. Add strict MDP-lite validation with:

```powershell
python examples/run_reviewer_smoke.py --with-mdp
```

The MDP smoke loads the bundled NoCong/Ref model and requires model hit = 1,
model miss = 0, and fallback = 0 in the generated decision log.

## 6. Retransmission policies

| Policy | Implementation meaning |
|---|---|
| NoRet | Never continue after a failed first attempt. |
| Classical | Retransmit whenever budget remains. |
| Nomikos | Literature-inspired/conceptual hard-deadline baseline; retransmit only when the predicted next arrival is still timely. It is not claimed as a complete reproduction of a published protocol. |
| UDRC | Retransmit when expected remaining-chain TCE utility gain exceeds lambda-scaled normalized retransmission cost. |
| MDP-lite | Simplified model-driven baseline using an empirical transition table and a short-horizon recursive value. It is not a full MDP scheduler. |

The release reviewer/formal path fails if MDP-lite is selected without a valid
model. Historical chain fallback is available only through the explicit advanced
option `--mdp_allow_chain_fallback`; do not use it when claiming model-driven results.

## 7. Paper configuration

The preserved S15 command logs support the following formal research configuration:

| Setting | Formal value |
|---|---|
| Scenarios | Ref, UrbMask, Tunnel |
| Loads | NoCong and Cong |
| Duration / timestep / seeds | 60 s / 0.1 s / seeds 1–15 |
| Message rate / maximum distance | 10 Hz / 200 m |
| TCE | D=20 ms, G=50 ms, beta=ln(20), gamma=2 |
| TX selection | mix, k=6, cross k=2 |
| UDRC | lambda=0.03, cost mode `delay_cbr` |
| MDP-lite | ret=2, threshold=0, cost scale=0.85, min samples=3, discount=1 |

These values are paper/research settings, not generic CLI defaults. They are
stored in `configs/paper_s15.json`. To inspect a protected formal command without
starting the large run:

```powershell
python examples/run_paper_configuration.py --scenario UrbMask --load Cong --policy udrc
```

The helper is dry-run by default. `--execute` explicitly starts the 15-seed run.

## 8. How to change the experiment

| Goal | Option | Meaning | Safe example |
|---|---|---|---|
| Scenario | `--scenarios` | Ref/UrbMask/Tunnel list | `--scenarios Ref` |
| Load | `--enable_congestion` | Omit for NoCong; include for Cong | `--enable_congestion` |
| Policy | `--retx_policy` | One of five policies | `--retx_policy udrc` |
| Retry budget | `--rets` | Extra attempts: 0, 1, or 2 | `--rets 2` |
| Seeds | `--seed_start`, `--n_seeds` | Consecutive NumPy seeds | `--seed_start 1 --n_seeds 3` |
| Duration | `--duration_s` | Generated trajectory duration | `--duration_s 10` |
| Message rate | `--msg_rate_hz` | Safety messages per second | `--msg_rate_hz 10` |
| Communication scope | `--max_distance_m` | Hard TX–RX output filter | `--max_distance_m 200` |
| TCE deadline | `--tce_deadline_ms` | Strictly timely boundary | `--tce_deadline_ms 20` |
| TCE grace | `--tce_grace_ms` | Partially useful interval | `--tce_grace_ms 50` |
| UDRC trade-off | `--udrc_lambda` | Cost weight | `--udrc_lambda 0.03` |
| Cost definition | `--policy_cost_mode` | delay/airtime/CBR/p_col mode | `--policy_cost_mode delay_cbr` |
| MDP model | `--mdp_model_path` | Empirical JSON table | `--mdp_model_path models/...json` |
| MDP cost | `--mdp_cost_scale` | Online scale on empirical cost | `--mdp_cost_scale 0.85` |

Scenario, load, seed count, duration, and output scope are routine experiment
parameters. Changing TCE D/G/beta/gamma, UDRC lambda/cost definition, MDP scale,
propagation, traffic, or congestion equations changes the scientific
configuration and must not be described as reproducing S15 without new evidence.

## 9. Outputs

Each run is stored in `workspace/results/runs/<run_id>/`:

- `raw/results_packets...csv`: one row per message/TX/RX outcome;
- `raw/results_retx_decisions...csv`: per-failure policy decisions and diagnostics;
- `tables/policy_summary...csv`: seed-aggregated Timely/PHY/TCE/policy metrics;
- `tables/policy_compare...csv`: collected policy summaries;
- `tables/tce_summary...csv`: overall study-aligned TCE decomposition;
- `tables/readiness...csv`: hidden-zone/separation suitability check;
- `figures/`: optional Python plots;
- `run_manifest.json`: cumulative stage/config/input/model provenance;
- `run_commands.txt`: local execution log (excluded from Git).

See [Output schema](docs/OUTPUT_SCHEMA.md).

## 10. Reproducing paper-style results

The complete S15 study comprises scenarios × loads × policies and 15 seeds. It
was generated as separate policy runs sharing scenario inputs, followed by policy
comparison and MATLAB plotting. Use `configs/paper_s15.json` and the dry-run formal
helper to construct explicit commands; do not use `latest` for formal analysis.

The approximately 40 GB historical raw outputs are not included. This repository
contains the source and six small empirical MDP-lite models needed to regenerate
new runs. Full reproduction requires substantial compute/storage and should not
be confused with the reviewer smoke.

## 11. MDP-lite models

Six JSON models are under `models/`: NoCong/Cong × Ref/UrbMask/Tunnel. Checksums
and provenance are in `models/MODEL_MANIFEST.csv`. They were built offline from
Classical ret=2 decision/packet data for seeds 1–15. The formal S15 runs recorded
model hit 1.0, model miss 0.0, and fallback 0.0 for all six combinations.

These models were not evaluated as an independent train/test generalization
study: the empirical tables and formal evaluations use the same seed range.
Select the model matching both scenario and load.

## 12. Computational notes

- TCE is O(1) time and O(1) auxiliary state per packet.
- Nomikos is O(1) per decision.
- UDRC is O(R) per decision for remaining retry budget R; R<=2 in S15.
- MDP-lite online lookup/value recursion is expected O(R) for a bounded set of dictionary keys.
- MDP table construction is offline and uses DataFrame sort/merge/group operations; conservatively O(N log N) time and O(N) memory for N source decision rows.

## 13. Model scope and limitations

The simulator uses analytical/stochastic system-level propagation, reception,
delay, and congestion abstractions. It is not a full protocol-level 802.11p,
C-V2X, or NR-V2X stack. It does not implement HARQ, SPS, realistic resource
scheduling, ns-3, OMNeT++, Veins, or SUMO.

TCE currently does not explicitly model AoI, inter-reception interval,
burstiness-aware utility, consecutive-loss utility, freshness history, or
application-state evolution. A fixed 10 Hz message source is not equivalent to
a burstiness-aware application model.

## 14. Reproducibility

Simulation, trajectory, building, and link-variation randomness uses explicit
NumPy seeds. The release stores relative model references and SHA-256 values in
the manifest. `requirements.txt` pins the validated Python environment. Formal
runs must use explicit run IDs, configuration, models, and seeds rather than
mtime/`latest` selection. See [Reproducibility](docs/REPRODUCIBILITY.md).
