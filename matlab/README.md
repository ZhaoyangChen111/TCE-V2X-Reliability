# MATLAB paper figures

`plot_s15_paper_figures.m` is the final seven-figure script retained from the
S15 + MDP study. It expects completed NoCong and Cong run directories containing
the `policy_compare` and `tce_by_distance` tables.

From this directory, for example:

```matlab
plot_s15_paper_figures('../workspace/results/runs', ...
    'S15_NoCong_RUN_ID', 'S15_Cong_RUN_ID')
```

The original experiments did not formally pin a MATLAB release. The release
candidate was inspected with a MATLAB R2025 executable available on the audit
machine, but the plotting script was not used to regenerate the formal figures
during release packaging.
