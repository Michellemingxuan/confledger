"""Dev-slice audits that must pass before the grid is collected (Sec. 4.1).

Step 1 -- export:  python experiments/scripts/dev_audit.py -c CFG -m gpt-4.1 export
  writes extraction_audit.csv (50 random raw responses + parsed answer) and
  paraphrase_audit.csv (paraphrase sets) with empty columns for a human to fill:
  `human_answer` (yes/no/unparseable) and `meaning_ok` (1/0).
Step 2 -- score:   python experiments/scripts/dev_audit.py -c CFG -m gpt-4.1 score
  reports extraction accuracy and paraphrase meaning preservation with Jeffreys
  intervals. Suggested gates: extraction >= 0.98, meaning preservation >= 0.95.
"""
import csv
import random

from confledger.utility.config import arg_parser, from_args, results_dir
from confledger.utility.llm import cache_path, read_responses

from confledger.utility.answer_parsing import parse_answer, parse_json_list
from confledger.metrics import jeffreys_interval

ap = arg_parser(__doc__)
ap.add_argument("action", choices=["export", "score"])
ap.add_argument("--n", type=int, default=50)
a = ap.parse_args()
cfg = from_args(a)
out = results_dir(cfg) / "audit"          # the response cache spans datasets
out.mkdir(parents=True, exist_ok=True)
mk = cfg["_models"][0]


if a.action == "export":
    ans = list(read_responses(cache_path(cfg["paths"]["cache"], f"target-{mk}"), "answer"))
    rng = random.Random(0)
    with open(out / "extraction_audit.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["response", "parsed", "human_answer"])
        for r, resp in rng.sample(ans, min(a.n, len(ans))):
            fmt = "ab" if "option A if the answer is No" in r["prompt"] else "yn"
            w.writerow([resp, parse_answer(resp, fmt) or "NONE", ""])
    paras = list(read_responses(cache_path(cfg["paths"]["cache"], "generator"), "paraphrase_gen"))
    with open(out / "paraphrase_audit.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["original_prompt_excerpt", "paraphrase", "meaning_ok"])
        for r, resp in rng.sample(paras, min(20, len(paras))):
            orig = r["prompt"].split("Question:")[-1].split("\n\n")[0].strip()
            for ptxt in (parse_json_list(resp) or [])[:3]:
                w.writerow([orig, ptxt, ""])
    print(f"wrote {out/'extraction_audit.csv'} and {out/'paraphrase_audit.csv'}; fill the last column")
else:
    ext = list(csv.DictReader(open(out / "extraction_audit.csv")))
    lab = [r for r in ext if r["human_answer"].strip()]
    ok = sum((r["parsed"].lower() == r["human_answer"].strip().lower()) or
             (r["parsed"] == "NONE" and r["human_answer"].strip().lower() == "unparseable")
             for r in lab)
    lo, hi = jeffreys_interval(ok, len(lab))
    print(f"extraction accuracy: {ok}/{len(lab)}  [{lo:.3f}, {hi:.3f}]")
    par = [r for r in csv.DictReader(open(out / "paraphrase_audit.csv")) if r["meaning_ok"].strip()]
    ok2 = sum(r["meaning_ok"].strip() == "1" for r in par)
    lo, hi = jeffreys_interval(ok2, len(par))
    print(f"paraphrase meaning preserved: {ok2}/{len(par)}  [{lo:.3f}, {hi:.3f}]")
