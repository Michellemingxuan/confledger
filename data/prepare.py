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
from data import load_items, make_split
from data.loaders import HF_SOURCES


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
sp = cfg["split"]
need = sp["n_dev"] + sp["n_eval"] + sp.get("n_cal", 0)

from datasets import load_dataset  # noqa: E402

for name in cfg["_datasets"]:
    out = results_dir(cfg, name) / "data"
    out.mkdir(parents=True, exist_ok=True)
    hf, conf, split = HF_SOURCES[name]
    ds = load_dataset(hf, conf, split=split)
    info = {"hf_id": hf, "config": conf, "split": split, "fingerprint": ds._fingerprint,
            **check(name, ds)}
    usable = info.get("unique_question_ids", info["rows"])
    assert usable >= need, f"{name}: {usable} usable rows < {need} needed"
    if name != "triviaqa":
        items = load_items(name, need, cfg["seed"])
        s = make_split(items, sp["n_dev"], sp["n_eval"], sp.get("batch_size"), cfg["seed"], sp.get("n_cal", 0))
        for part, its in (("dev", s.dev), ("eval", s.eval), ("cal", s.cal)):
            if not its:
                continue
            with open(out / f"{name}__{part}.jsonl", "w") as fh:
                for it in its:
                    fh.write(json.dumps({"item_id": it.item_id, "question": it.question,
                                         "gold": it.gold}) + "\n")
            info[f"{part}_labels"] = dict(Counter(it.gold for it in its))
        assert len({it.item_id for it in items}) == len(items), "duplicate item ids"
    print(f"{name}: {json.dumps(info)}")
    (out / "manifest.json").write_text(json.dumps({name: info}, indent=2))
    print(f"wrote {out / 'manifest.json'}")
