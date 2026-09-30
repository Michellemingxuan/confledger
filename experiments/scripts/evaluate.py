"""Run the Sec. 4 protocol on every selected cell (no API calls) and write the
per-cell reports plus <results>/summary.json (Holm-corrected confirmatory tests).

    python experiments/scripts/evaluate.py -c configs/smoke.yaml
"""
import json

from confledger.evaluation_protocol import evaluate_config
from confledger.utility.config import arg_parser, from_args

print(json.dumps(evaluate_config(from_args(arg_parser(__doc__).parse_args())), indent=2))
