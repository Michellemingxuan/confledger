"""End-to-end on the simulator: collection, replay accounting, protocol, evidence ledger."""
import pytest

from confledger.collection import CollectConfig, Collector, target_calls_per_item
from data import make_split, synthetic_items
from confledger.evaluation_protocol import run_protocol
from confledger.utility.llm import CachedClient
from confledger.utility.llm.simulated_model import SimulatedBackend
from confledger.methods import CELConfig, ledger_statement, cel, self_consistency


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("sim")
    mk = lambda tag: CachedClient(SimulatedBackend(seed=1, name=tag), d / f"{tag}.sqlite")
    t, g, j = mk("t"), mk("g"), mk("j")
    items = synthetic_items("triviaqa", 90, seed=3)
    split = make_split(items, n_dev=10, n_eval=80, batch_size=20, seed=0)
    col = Collector(t, g, j, CollectConfig(workers=4))
    coll = {cd.item_id: cd for cd in col.collect_all(split.eval, d / "c.jsonl")}
    return {"t": t, "g": g, "j": j, "coll": coll, "split": split, "items": items, "dir": d}


def test_collection_shape_and_cache(world):
    cd = next(iter(world["coll"].values()))
    assert len(cd.fresh) == 12 and len(cd.same) == 16
    assert cd.reuse[0][0] is not None and cd.reuse[1][0].answer == cd.fresh[1].answer
    before = world["t"].misses
    Collector(world["t"], world["g"], world["j"], CollectConfig()).collect(world["split"].eval[0])
    assert world["t"].misses == before                     # fully cached re-run


def test_call_accounting(world):
    for cd in world["coll"].values():
        s = cel(cd, CELConfig())
        assert s.calls == (16 if s.survived else 4)          # para-gen + 3 + 6 + 3 probes + se + chal + verb
        assert self_consistency(cd, 8).calls == 8


def test_protocol_runs(world):
    res = run_protocol(world["coll"], world["split"].batches, k=5, n_curve_seeds=2)
    assert res["matched_budget"] >= 2
    assert abs(res["shapley"]["efficiency_check"]) < 1e-9
    assert set(res["methods"]) >= {"cel", "self_consistency", "verbalized", "random"}
    assert 0 <= res["methods"]["cel"]["mean_prec"] <= 1
    assert res["methods"]["self_consistency_deploy"]["mean_calls"] == 3      # deployment baseline
    assert "sc:3" in res["budget_curve"]
    # signal effects: leave-one-out for all five signals, Shapley with CI on prec@k
    sig = {"L", "flip", "se", "chal", "verb"}
    assert set(res["leave_one_out"]) == sig
    assert res["leave_one_out"]["se"]["calls_saved"] == pytest.approx(1.0 * res["methods"]["cel"]["survival"])
    sh = res["shapley"]
    for j in sig:
        lo, hi = sh["phi_ci"][j]
        assert lo - 1e-12 <= sh["phi"][j] <= hi + 1e-12


def test_ledger_statement_matches_pipeline(world):
    for cd in world["coll"].values():
        sc, st = cel(cd), ledger_statement(cd)
        if sc.survived:
            assert st["balance_nats"] == pytest.approx(sc.signals["llr"])
            assert st["calls"] == 9 + len(sc.probe_passes)
        else:
            assert st["calls"] == 3 and not st["certified"]


def test_protocol_refuses_small_collection(world):
    import copy
    small = copy.deepcopy(world["coll"])
    for cd in small.values():
        cd.fresh, cd.same = cd.fresh[:3], cd.same[:3]
    with pytest.raises(ValueError, match="fresh-paraphrase draws.*same-prompt draws"):
        run_protocol(small, world["split"].batches, k=5, n_curve_seeds=2)


def test_verbal_per_draw(tmp_path):
    mk = lambda tag: CachedClient(SimulatedBackend(seed=1, name=tag), tmp_path / f"{tag}.sqlite")
    item = synthetic_items("triviaqa", 1, seed=3)[0]
    t, cc = mk("t"), CollectConfig(verbal_per_draw=True)
    cd = Collector(t, mk("g"), mk("j"), cc).collect(item)
    assert all(d.confidence is not None for d in cd.fresh + cd.same if d.answer is not None)  # malformed draws: neither
    assert cd.verbal_conf == cd.same[0].confidence
    assert t.misses == target_calls_per_item(vars(cc)) == 12 + 16 + 6 + 4 + 1   # no verbal call
    s = cel(cd)
    if s.survived:
        assert s.calls == 15 and s.signals["verb"] > 0                    # verbal from the draws


def test_budget_allocated_cel_k(world):
    from confledger.methods import Allocation, cel_alloc
    alloc = Allocation(k1=2, extra=1, flips=("neg", "relabel"), chal=True, se=True)
    for cd in list(world["coll"].values())[:20]:
        c = cel_alloc(cd, alloc)
        # eliminated: the 2 screen draws; survivor: + 1 draw + 2 flips + challenge + judge
        assert c.calls == (2 + 1 + 2 + 1 + 1 if c.survived else 2)
    # allocation chosen on a "dev" slice (first eval batch), evaluated on the rest
    dev_ids = world["split"].batches[0]
    res = run_protocol(world["coll"], world["split"].batches[1:], k=5, n_curve_seeds=2,
                       budget_ks=(3, 5), dev_coll=world["coll"], dev_batches=[dev_ids])
    for B in ("3", "5"):
        b = res["budget_matched"]["by_k"][B]
        assert b["dev_calls"] <= int(B) + 1e-9                    # expected calls within budget
        assert b["self_consistency"]["calls"] == int(B)            # exactly B draws
    assert "celk:3" in res["budget_curve"] and res["budget_matched"]["metric"] == "prec@k"

def test_adaptive_cel_nests_in_allocation_space(world):
    from confledger.methods import allocation_space, cel_alloc, cel_as_allocation
    base = CELConfig()
    a = cel_as_allocation(base, verbal_per_draw=False)
    assert a in allocation_space(verbal_per_draw=False)                 # max_extra reaches s2
    for cd in world["coll"].values():
        c, n = cel(cd, base), cel_alloc(cd, a)
        assert (c.answer, c.survived, c.calls) == (n.answer, n.survived, n.calls)
        assert (c.signals.get("m"), c.signals.get("k")) == ((n.signals["m"], n.signals["k"]) if c.survived
                                                            else (None, None))


def test_mode_linkage(world):
    dev_ids = world["split"].batches[0]
    res = run_protocol(world["coll"], world["split"].batches[1:], k=5, n_curve_seeds=2, budget_ks=(3,),
                       dev_coll=world["coll"], dev_batches=[dev_ids], linkage_ks=(3, 8, 16))
    L = res["linkage"]
    assert set(res["budget_matched"]["by_k"]) == {"3"}                  # linkage budgets not in the SC table
    ke = [L["derived"][B]["k_eff"] for B in ("3", "8", "16") if B in L["derived"]]
    assert ke == sorted(ke)                                             # more budget, no less evidence
    for d in L["derived"].values():
        assert d["dev_calls"] <= 16 + 1e-9 and 0 <= d["certified_exact"] <= 1
    thr = L["certification_budget"]
    assert thr["model"] is None or thr["exact"] is None or thr["model"] <= thr["exact"] + 1e-9
    assert L["nested"]["calls"] == pytest.approx(L["adaptive"]["cel"]["calls"])
