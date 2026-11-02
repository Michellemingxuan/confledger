"""Re-group the rationales of existing collections with the config's judge, without new
model-under-test calls (decisions.md #33): every draw's `cluster` becomes the judge's group.
Writes a .bak copy of each collection first. Records already grouped by this judge are skipped.

    python experiments/scripts/recluster.py -c configs/smoke.yaml
"""
import shutil
import time

from confledger.collection import api_cluster_file, collection_path, make_judge, nli_cluster_file
from confledger.utility.config import arg_parser, cells, from_args
from data.splits import load_split

a = arg_parser(__doc__).parse_args()
cfg = from_args(a)
judge = make_judge(cfg["judge"], cfg["paths"]["cache"], "judge")
for ds, mk in cells(cfg):
    items, _ = load_split(cfg, ds)
    question = {it.item_id: it.question for it in items}
    for suffix in ("", "__dev", "__cal", "__reseed"):
        path = collection_path(cfg, ds, mk, suffix)
        if not path.exists():
            continue
        if not path.with_suffix(".jsonl.bak").exists():
            shutil.copy(path, path.with_suffix(".jsonl.bak"))
        t0 = time.time()
        if hasattr(judge, "cluster_many"):
            n, failed = nli_cluster_file(path, judge, question), 0
        else:
            n, failed = api_cluster_file(path, judge, question, cfg["collect"]["workers"])
        print(f"{path.name}: {n} records clustered, {failed} failed, {time.time() - t0:.0f}s")
