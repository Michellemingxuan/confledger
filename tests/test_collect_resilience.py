"""Collection resilience for the long private-environment run: a refused request is not
retried, one failed item does not block the others, and a systematic fault stops early."""
import json
import time

import pytest

from confledger import collection
from confledger.collection import CollectConfig, Collector
from confledger.utility.llm import CachedClient, Rejected, Request
from confledger.utility.llm import response_cache
from confledger.utility.records import CaseData, Item


class _Backend:
    name = "fake"

    def __init__(self, exc):
        self.exc, self.calls = exc, 0

    def generate(self, req):
        self.calls += 1
        raise self.exc


def test_rejected_is_not_retried_but_other_errors_are(tmp_path, monkeypatch):
    monkeypatch.setattr(response_cache.time, "sleep", lambda s: None)
    b = _Backend(Rejected("safechain blocked"))
    with pytest.raises(Rejected):
        CachedClient(b, tmp_path / "a.sqlite", max_retries=5).generate(Request("q"))
    assert b.calls == 1                                   # refused for good: one attempt
    b2 = _Backend(RuntimeError("503"))
    with pytest.raises(RuntimeError):
        CachedClient(b2, tmp_path / "b.sqlite", max_retries=3).generate(Request("q"))
    assert b2.calls == 3                                  # transient errors still back off and retry


class _Dummy:
    misses = 0


class _Col(Collector):
    def __init__(self, bad, **kw):
        super().__init__(_Dummy(), _Dummy(), _Dummy(), CollectConfig(workers=1, **kw))
        self.bad, self.tried = set(bad), []

    def collect(self, item):
        self.tried.append(item.item_id)
        time.sleep(0.05)                                  # a real call takes seconds; let the loop react
        if item.item_id in self.bad:
            raise Rejected("blocked")
        return CaseData(item.item_id, "d", "yes", [], [], [], {})


def _items(n):
    return [Item(f"i{k}", "d", "q", "yes") for k in range(n)]


def test_failed_item_is_recorded_and_skipped_then_retried_on_rerun(tmp_path):
    out = tmp_path / "c.jsonl"
    col = _Col(bad={"i2"})
    with pytest.raises(RuntimeError, match="1 of 5 items failed"):
        col.collect_all(_items(5), out, progress=False)
    kept = [json.loads(l)["item_id"] for l in out.read_text().splitlines()]
    assert sorted(kept) == ["i0", "i1", "i3", "i4"]       # the other items were not blocked
    assert list(json.loads(out.with_suffix(".errors.json").read_text())) == ["i2"]
    again = _Col(bad=set())                               # fault fixed: only the failed item is redone
    got = again.collect_all(_items(5), out, progress=False)
    assert again.tried == ["i2"] and [c.item_id for c in got] == [f"i{k}" for k in range(5)]


def test_systematic_failure_stops_early(tmp_path):
    col = _Col(bad={f"i{k}" for k in range(50)}, max_errors=3)
    with pytest.raises(RuntimeError, match="items failed"):
        col.collect_all(_items(50), tmp_path / "c.jsonl", progress=False)
    assert len(col.tried) <= 5                            # not 50 attempts at 30 s of retries each


def test_progress_bar_runs_with_tqdm(tmp_path, capsys):
    pytest.importorskip("tqdm")
    col = _Col(bad=set())
    col.collect_all(_items(3), tmp_path / "c.jsonl", desc="demo", progress=True)
    err = capsys.readouterr().err
    assert "demo" in err and "3/3" in err


class _ParaCol(Collector):
    def __init__(self, bad, **kw):
        super().__init__(_Dummy(), _Dummy(), _Dummy(), CollectConfig(workers=1, **kw))
        self.bad, self.tried = set(bad), []

    def paraphrases(self, item):
        self.tried.append(item.item_id)
        time.sleep(0.05)
        if item.item_id in self.bad:
            raise Rejected("blocked")
        return ["p"]


def test_paraphrase_all_skips_failed_items_and_stops_on_systematic_failure(tmp_path):
    col = _ParaCol(bad={"i1"})
    errs = col.paraphrase_all(_items(4), progress=False)
    assert list(errs) == ["i1"] and sorted(col.tried) == ["i0", "i1", "i2", "i3"]
    allbad = _ParaCol(bad={f"i{k}" for k in range(50)}, max_errors=3)
    assert len(allbad.paraphrase_all(_items(50), progress=False)) >= 3 and len(allbad.tried) <= 5


def test_paraphrase_all_progress_bar(capsys):
    pytest.importorskip("tqdm")
    _ParaCol(bad=set()).paraphrase_all(_items(3), desc="paras", progress=True)
    err = capsys.readouterr().err
    assert "paras" in err and "3/3" in err
