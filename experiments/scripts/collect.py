"""Collect every selected cell of a config. The ONLY step that spends API calls.

    python experiments/scripts/collect.py -c configs/smoke.yaml --reseed-batches 0 -v

Resumable: finished items are skipped and cached responses cost nothing. Refuses to
start while any model id in the selected cells is a <placeholder>.
"""
import logging

from confledger.collection import collect_config
from confledger.utility.config import arg_parser, from_args

ap = arg_parser(__doc__)
ap.add_argument("--reseed-batches", type=int, default=0, help="re-collected batches; 0: stability is re-drawn from cached draws (decisions.md #29)")
ap.add_argument("-v", "--verbose", action="store_true")
ap.add_argument("--no-progress", action="store_true", help="no progress bar (it writes carriage returns, which clutter a log file)")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                    format="%(levelname)s %(name)s: %(message)s")
collect_config(from_args(a), a.reseed_batches, progress=not a.no_progress)
