"""Hash-keyed SQLite cache of every raw response (Sec. 4.3).

Key = (backend name, prompt, temperature, sample index). Re-running any policy,
ablation, budget curve or Shapley coalition hits the cache: zero marginal cost.
Thread-safe; logs cache hits/misses for the budget report.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from .request import Backend, Request


class CachedClient:
    def __init__(self, backend: Backend, cache_path: str | Path, max_retries: int = 5):
        self.backend = backend
        self.max_retries = max_retries
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, "
                         "backend TEXT, request TEXT, response TEXT, ts REAL)")
        self._db.commit()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return self.backend.name

    def _key(self, req: Request) -> str:
        key = [self.backend.name, req.prompt, round(req.temperature, 4), req.sample, req.max_tokens]
        if req.logprobs:                         # only then, so existing keys stay valid
            key.append("logprobs")
        payload = json.dumps(key, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def generate(self, req: Request) -> str:
        key = self._key(req)
        with self._lock:
            row = self._db.execute("SELECT response FROM responses WHERE key=?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return row[0]
        self.misses += 1
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                text = self.backend.generate(req)
                break
            except Exception:  # provider errors, rate limits
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        rec = json.dumps({"prompt": req.prompt, "temperature": req.temperature,
                          "sample": req.sample, "kind": req.kind}, ensure_ascii=False)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?)",
                             (key, self.backend.name, rec, text, time.time()))
            self._db.commit()
        return text


def cache_path(cache_dir: str | Path, tag: str) -> Path:
    """One SQLite file per client: 'generator', 'target-<model>', 'judge-<key>'."""
    return Path(cache_dir) / f"{tag}.sqlite"


def cached_client(spec: dict, cache_dir: str | Path, tag: str) -> CachedClient:
    """A config model entry ({provider, model, ...}) behind its response cache."""
    from .backends import make_backend  # noqa: PLC0415
    return CachedClient(make_backend(spec), cache_path(cache_dir, tag))


def read_responses(cache_path: str | Path, kind: str):
    """Yield (request record, raw response) for every cached call of one kind."""
    con = sqlite3.connect(str(cache_path))
    for req, resp in con.execute("SELECT request, response FROM responses"):
        r = json.loads(req)
        if r.get("kind") == kind:
            yield r, resp
