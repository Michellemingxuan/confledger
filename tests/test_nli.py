"""Greedy bidirectional-entailment grouping of rationales (utility/nli.py), with the NLI
model replaced by a stub so the test needs no weights."""
from confledger.utility.nli import NLIClusterer


def stub(entails):
    c = object.__new__(NLIClusterer)
    c.with_question = False
    c._entails = lambda pairs: [entails(p, h) for p, h in pairs]
    return c


def test_mutual_entailment_groups():
    same_reason = lambda p, h: p[0] == h[0]                  # first letter = the "reason"
    c = stub(same_reason)
    assert c.cluster("q", ["apple", "avocado", "banana", "apricot", "", "blue"]) == [0, 0, 1, 0, None, 1]


def test_lockstep_matches_one_at_a_time_and_drops_confidence_line():
    c = stub(lambda p, h: p[0] == h[0])
    pools = [["apple\nConfidence: 90", "avocado", "berry"], ["cat", "dog", "cow", "dove"]]
    assert c.cluster_many(["q", "q"], pools) == [c.cluster("q", p) for p in pools] == [[0, 0, 1], [0, 1, 0, 1]]
    assert c.cluster("q", ["kiwi\nConfidence: 90", "kiwi\nConfidence: 60"]) == [0, 0]


def test_one_way_entailment_is_not_enough():
    c = stub(lambda p, h: len(p) >= len(h))                  # longer entails shorter, not back
    assert c.cluster("q", ["ab", "abc"]) == [0, 1]
    assert c.cluster("q", ["ab", "ab"]) == [0, 0]            # identical text needs no model call


def test_api_judge_labels_and_malformed_reply():
    from confledger.collection import api_labels

    class Judge:
        def __init__(self, reply): self.reply = reply
        def generate(self, req): return self.reply

    assert api_labels(Judge("[0, 0, 1]"), "q", ["a", "b", "c"]) == [0, 0, 1]
    assert api_labels(Judge("[0, 1]"), "q", ["a", "b", "c"]) is None
    assert api_labels(Judge("no list"), "q", ["a"]) is None
