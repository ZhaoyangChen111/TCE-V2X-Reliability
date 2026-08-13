# Output schema

## Packet output

Identity fields include scenario, retry budget, policy/tag, seed, message/time,
TX/RX IDs, and distance. Outcome fields include `success` (raw packet-deadline
success), `success_phy`, `late`, reason, attempts, and delay. Channel/congestion
fields include link state, blockage, CBR, collision probability and congestion
delay. Downstream TCE uses `success_phy`, not raw `success`, as its primary source.

## Decision log

Each decision row records attempt state, predicted delay/utility/success,
`cost_ci`, cost components, UDRC chain gain/cost/score, and final retransmit/stop.
MDP rows additionally expose exact/coarse/global hit, miss, fallback, model
samples, lookup key/scope, value terms, cost scale, and decision source.

## Summaries

- `policy_summary`: mean and normal-approximation 95% CI across seed metrics.
- `policy_compare`: first row from matching policy summaries plus source filename.
- `tce_summary`: Timely, PHY, TCE, late count/ratio, and partial gain.
- `readiness`: research separation/hidden-zone checks, not correctness certification.

## Manifest

The release manifest is create-if-absent, cumulatively updated, deep-merged, and
atomically replaced. It records stages, command/config, load/scenario/policy,
seeds, effective TCE, model path/hash, input path/hash, and stage outputs.
