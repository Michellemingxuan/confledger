"""Make every generator output a config needs (TriviaQA distractors, paraphrases) into the
shared generator cache, before any model is collected. Run once on one machine and commit
`paths.generator_cache`, so all machines and models get identical items (decisions.md #35).

    python experiments/scripts/pregenerate.py -c configs/pilot_1000.yaml
"""
import logging
from concurrent.futures import ThreadPoolExecutor

from confledger.collection import CollectConfig, Collector
from confledger.utility.config import arg_parser, from_args
from data.splits import generator_client, load_split

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
cfg = from_args(arg_parser(__doc__).parse_args())
gen = generator_client(cfg)
col = Collector(None, gen, None, CollectConfig(**cfg["collect"]))
for ds in cfg["_datasets"]:
    _, split = load_split(cfg, ds, gen)
    items = split.dev + split.eval + split.cal
    with ThreadPoolExecutor(cfg["collect"].get("workers", 8)) as ex:
        list(ex.map(col.paraphrases, items))
    print(f"{ds}: {len(items)} items; generator calls {gen.misses} new, {gen.hits} cached; "
          f"short paraphrase sets so far {col.failures['paraphrase']}")
