# Reproducibility

## Environment and randomness

The validated versions are pinned in `requirements.txt`. Simulation and scenario
generators use explicit NumPy generators; the link-variation field derives stable
per-link/block seeds using SHA-256. Run IDs affect storage paths, not random draws.

## Models

Verify `models/MODEL_MANIFEST.csv` before formal MDP-lite work. A model must match
scenario and load. Strict release execution rejects a missing model and disables
chain fallback unless the advanced debug flag is explicit.

## Workflow

1. Run the basic and optional MDP reviewer smoke.
2. Inspect `configs/paper_s15.json`.
3. Use `examples/run_paper_configuration.py` without `--execute` to inspect each command.
4. Allocate storage before a formal run.
5. Preserve the resulting manifest, commands, model/input hashes, raw data, and summaries together.
6. Run `compare_policies_C.py` with an explicit run ID after the required policy summaries exist.
7. Pass explicit NoCong/Cong run IDs to the MATLAB script.

The historical S15 raw data was about 40 GB and is intentionally omitted from
GitHub. Models and source are included; generated `workspace/` data is ignored.

## Known boundary

The empirical MDP-lite tables and formal evaluation used the same seed range
1–15. They verify model-driven execution, not independent train/test generalization.
