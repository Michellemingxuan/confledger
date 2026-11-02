"""Free text signals, confidence under negation, and the gray-box answer probability."""
import math
from types import SimpleNamespace as NS

import pytest

from confledger.collection import CollectConfig, Collector
from confledger.methods import Allocation, cel_alloc
from confledger.score_model import FEATURES, GRAY_BOX, fit_score_model
from confledger.utility.answer_parsing import (answer_token_prob, has_hedge, p_yes_trailer,
                                               rationale_words, split_p_yes)
from confledger.utility.llm import CachedClient, Request
from confledger.utility.llm.simulated_model import SimulatedBackend
from data import synthetic_items


def test_answer_token_prob_normalises_yes_no():
    tok = lambda t, lp, top=None: NS(token=t, logprob=lp, top_logprobs=top)
    content = [tok("Answer", -0.1), tok(":", -0.1),
               tok(" Yes", math.log(0.6), [tok(" Yes", math.log(0.6)), tok(" No", math.log(0.2)),
                                          tok("Maybe", math.log(0.2))]),
               tok("\n", -0.1)]
    assert answer_token_prob(content) == pytest.approx(0.75)        # 0.6 / (0.6 + 0.2)
    assert answer_token_prob([tok("Yes", -0.1)]) is None             # no "Answer" before it
    assert answer_token_prob(None) is None


def test_trailer_round_trip_and_text_signals():
    text, p = split_p_yes("Answer: Yes\nExplanation: x" + p_yes_trailer(0.8125))
    assert text == "Answer: Yes\nExplanation: x" and p == pytest.approx(0.8125)
    assert split_p_yes("Answer: No") == ("Answer: No", None)
    assert rationale_words("It is probably the capital.") == 5
    assert has_hedge("It is probably the capital.") and not has_hedge("It is the capital.")


def test_cache_key_unchanged_without_logprobs(tmp_path):
    c = CachedClient(SimulatedBackend(), tmp_path / "c.sqlite")
    r = Request("q", 0.7, sample=1)
    assert c._key(r) != c._key(Request("q", 0.7, sample=1, logprobs=True))
    import hashlib, json
    old = hashlib.sha256(json.dumps([c.name, "q", 0.7, 1, 300]).encode()).hexdigest()
    assert c._key(r) == old                                           # existing caches stay valid


def test_collected_signals_reach_the_score(tmp_path):
    mk = lambda tag: CachedClient(SimulatedBackend(seed=1, name=tag), tmp_path / f"{tag}.sqlite")
    col = Collector(mk("t"), mk("g"), mk("j"), CollectConfig(verbal_per_draw=True, logprobs=True, workers=2))
    cases = [col.collect(it) for it in synthetic_items("triviaqa", 40, seed=5)]
    cd = cases[0]
    assert all(d.p_yes is not None for d in cd.fresh + cd.same if d.answer is not None)
    assert cd.probe_conf["neg"] is not None
    a = Allocation(2, False, 1, ("neg",), False, False)
    scored = [cel_alloc(c, a) for c in cases]
    surv = [s for s in scored if s.survived]
    assert surv and all({"length", "hedge", "p_first"} <= set(s.signals) for s in surv)
    assert any("neg_drop" in s.signals for s in surv)
    m = fit_score_model(scored, [s.correct for s in scored], drop=GRAY_BOX)
    assert abs(m.coef[FEATURES.index("p_first")]) < 1e-6             # gray-box held out by default
