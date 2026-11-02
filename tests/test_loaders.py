"""Dataset loaders against a fake HF dataset (no network)."""
import sys
import types

from data.loaders import load_triviaqa


class _FakeDS:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self.rows]
        return self.rows[key]


def test_triviaqa_dedups_question_ids(monkeypatch):
    # rc.nocontext lists each question twice (web + wiki) under one question_id
    rows = [{"question": f"q{i}", "question_id": f"id{i}",
             "answer": {"value": f"a{i}", "aliases": [f"a{i}"]}}
            for i in range(30) for _ in range(2)]
    fake = types.SimpleNamespace(load_dataset=lambda *a, **k: _FakeDS(rows))
    monkeypatch.setitem(sys.modules, "datasets", fake)
    items = load_triviaqa(30, seed=0, distractor_fn=None)
    ids = [it.item_id for it in items]
    assert len(ids) == 30 and len(set(ids)) == 30


def test_pilot_sets_are_the_first_items_of_the_full_sets(monkeypatch):
    import data.splits as S
    rows = [{"question": f"q{i}", "question_id": f"id{i}", "answer": {"value": f"a{i}", "aliases": [f"a{i}"]}}
            for i in range(50)]
    monkeypatch.setattr(S, "load_source", lambda ds: _FakeDS(rows))
    base = {"seed": 0, "generator": {}, "paths": {"cache": "unused"}}
    frac = {"dev": 0.2, "cal": 0.2, "eval": 0.6}
    full = S.split_ids({**base, "split": {"frac": frac}}, "triviaqa")
    pilot = S.split_ids({**base, "split": {"frac": frac, "n_dev": 3, "n_eval": 7, "n_cal": 2}}, "triviaqa")
    assert [len(full[0]), len(full[1][0]), len(full[2])] == [10, 30, 10]
    assert pilot[0] == full[0][:3] and pilot[1][0] == full[1][0][:7] and pilot[2] == full[2][:2]
    assert len(set(full[0]) | set(full[1][0]) | set(full[2])) == 50


def test_items_do_not_depend_on_which_others_are_built():
    from data.loaders import build_items
    rows = [{"question": f"q{i}", "question_id": f"id{i}", "answer": {"value": f"a{i}", "aliases": [f"a{i}"]}}
            for i in range(20)]
    ds = _FakeDS(rows)
    every = {it.item_id: it for it in build_items("triviaqa", ds, range(20), seed=0)}
    some = build_items("triviaqa", ds, [3, 11, 17], seed=0)
    assert all(it.question == every[it.item_id].question and it.gold == every[it.item_id].gold for it in some)
