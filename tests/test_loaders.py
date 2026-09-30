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
