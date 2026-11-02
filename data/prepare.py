"""Download and check the raw datasets, and freeze the item manifests. No API calls.

    python -m data.prepare -c configs/default.yaml

For each dataset: downloads from HF (cached under ~/.cache/huggingface), checks the
field types the loaders rely on, and records the dataset fingerprint. StrategyQA
and MMLU items are built with the real loaders + split and written to
<results>/<ds>/<config>/data/<ds>__{dev,eval,cal}.jsonl for inspection, next to a
manifest.json per dataset. TriviaQA items need the
generator model for distractors (an API call), so they are built at collect time;
a distractor-free preview would consume the RNG differently and not match.
"""
import json
from collections import Counter

from confledger.utility.config import arg_parser, from_args, results_dir
from data import load_split
from data.loaders import HF_SOURCES, load_source, usable_rows
from data.splits import set_sizes


def check(name: str, ds) -> dict:
    if name == "triviaqa":
        a = ds[0]["answer"]
        assert isinstance(a["value"], str) and isinstance(a["aliases"], list), a
        qids = ds["question_id"]
        return {"rows": len(ds), "unique_question_ids": len(set(qids))}
    if name == "strategyqa":
        assert all(isinstance(x, bool) for x in ds["answer"]), "answer must be bool"
        return {"rows": len(ds), "answer_balance": dict(Counter(map(str, ds["answer"])))}
    assert all(isinstance(x, int) for x in ds["answer"]), "answer must be int"
    assert all(len(c) == 4 for c in ds["choices"]), "expected 4 choices"
    return {"rows": len(ds)}


a = arg_parser(__doc__).parse_args()
cfg = from_args(a)

for name in cfg["_datasets"]:
    out = results_dir(cfg, name) / "data"
    out.mkdir(parents=True, exist_ok=True)
    ds = load_source(name)
    hf, conf, split = HF_SOURCES[name]
    info = {"hf_id": hf, "config": conf, "split": split, "fingerprint": ds._fingerprint,
            **check(name, ds), "distinct_questions": len(usable_rows(name, ds))}
    full, used = set_sizes(cfg["split"], info["distinct_questions"])
    assert sum(full.values()) <= info["distinct_questions"], f"{name}: split {full} exceeds the source"
    info |= {"full_sets": full, "used_sets": used}
    if name != "triviaqa":                      # TriviaQA items need the generator (collect time)
        _, s = load_split(cfg, name)
        for part, its in (("dev", s.dev), ("eval", s.eval), ("cal", s.cal)):
            if not its:
                continue
            with open(out / f"{name}__{part}.jsonl", "w") as fh:
                for it in its:
                    fh.write(json.dumps({"item_id": it.item_id, "question": it.question,
                                         "gold": it.gold}) + "\n")
            info[f"{part}_labels"] = dict(Counter(it.gold for it in its))
    print(f"{name}: {json.dumps(info)}")
    (out / "manifest.json").write_text(json.dumps({name: info}, indent=2))
    print(f"wrote {out / 'manifest.json'}")
