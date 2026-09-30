"""API calls a collection will cost, by model role, before spending anything.

    python experiments/scripts/estimate.py -c configs/smoke.yaml --reseed-batches 0
"""
from confledger.collection import estimate_calls
from confledger.utility.config import arg_parser, from_args

ap = arg_parser(__doc__)
ap.add_argument("--reseed-batches", type=int, default=2, help="re-drawn batches (Jaccard check)")
a = ap.parse_args()
rows = estimate_calls(from_args(a), a.reseed_batches)
print(f"{'dataset':12s} {'model':8s} {'items':>6s} {'target':>9s} {'judge':>7s} {'generator':>10s} {'total':>9s}")
for r in rows:
    tot = r["target"] + r["judge"] + r["generator"]
    print(f"{r['dataset']:12s} {r['model']:8s} {r['items']:>6d} {r['target']:>9,d} {r['judge']:>7,d} "
          f"{r['generator']:>10,d} {tot:>9,d}")
T = {k: sum(r[k] for r in rows) for k in ("target", "judge", "generator")}
print(f"{'TOTAL':28s} {T['target']:>9,d} {T['judge']:>7,d} {T['generator']:>10,d} {sum(T.values()):>9,d}")
