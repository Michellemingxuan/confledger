"""Make every generator output a config needs (TriviaQA distractors, paraphrases) into the
shared generator cache, before any model is collected. Run once on one machine and commit
`paths.generator_cache`, so all machines and models get identical items (decisions.md #35).

    python experiments/scripts/pregenerate.py -c configs/full_gpt41.yaml

Resumable: cached outputs cost nothing, so re-run after any stop. Shows a progress bar for the
distractors (TriviaQA) and for the paraphrases of each dataset. A failed item is skipped and
listed at the end (the exit status is then 1); after `collect.max_errors` failures it stops.
"""
import logging

from confledger.collection import CollectConfig, Collector
from confledger.utility.config import arg_parser, from_args
from data.splits import generator_client, load_split

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
ap = arg_parser(__doc__)
ap.add_argument("--no-progress", action="store_true", help="no progress bar (it writes carriage returns, which clutter a log file)")
a = ap.parse_args()
cfg = from_args(a)
gen = generator_client(cfg)
col = Collector(None, gen, None, CollectConfig(**cfg["collect"]))
failed = {}
for ds in cfg["_datasets"]:
    _, split = load_split(cfg, ds, gen)                       # TriviaQA distractors are made here
    items = split.dev + split.eval + split.cal
    errors = col.paraphrase_all(items, f"{ds} paraphrases", progress=not a.no_progress)
    failed.update({f"{ds}/{k}": v for k, v in errors.items()})
    print(f"{ds}: {len(items)} items; generator calls {gen.misses} new, {gen.hits} cached; "
          f"short paraphrase sets so far {col.failures['paraphrase']}; failed items {len(errors)}")
    if len(errors) >= col.cfg.max_errors:
        break
if failed:
    for k, v in list(failed.items())[:10]:
        print(f"FAILED {k}: {v}")
    raise SystemExit(f"{len(failed)} items failed; the rest is cached. Fix the cause and run again.")
