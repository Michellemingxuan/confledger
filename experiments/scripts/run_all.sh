#!/usr/bin/env bash
# Full pipeline for one config. Usage (from anywhere):
#   experiments/scripts/run_all.sh configs/smoke.yaml
#   experiments/scripts/run_all.sh configs/simulation.yaml      # dry run, no API keys
# collect.py is the only API spend (resumable); everything after it re-scores the cache.
set -euo pipefail
CFG="${1:-configs/default.yaml}"
cd "$(dirname "$0")/../.."
S=experiments/scripts
python $S/estimate.py -c "$CFG"
python $S/collect.py -c "$CFG" -v
python $S/evaluate.py -c "$CFG" > /dev/null      # per-cell protocol + reports + summary.json
for exp in sprt_characteristics main_ladder budget_curves calibration mode_linkage evidence_exchange monotonicity signal_effects signal_importance robustness; do
  python "$S/$exp.py" -c "$CFG"
done
echo "Paper tables and figures: see experiments/results/all/<config name>/paper/."
