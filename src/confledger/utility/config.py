"""Experiment configs: load one, check it before spending, and log every run.

A config (configs/*.yaml) is one run. Models under test are a list identified by
their model id; one `judge` (optionally an `alt_judge`) serves all of them.
`_datasets` / `_models` hold the cells selected on the command line. Relative paths resolve against the working
directory, so run everything from the repo root.

Outputs are grouped by dataset: <results>/<dataset>/<config name>/ holds one dataset's
collections and per-cell reports; <results>/all/<config name>/ holds what spans datasets
(summary.json, paper/, run_log.jsonl).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Iterator

import yaml


def load_env(path: str | Path = ".env") -> list[str]:
    """Export KEY=VALUE lines from a .env file (API keys) into the environment, without
    overriding variables already set in the shell. Returns the keys it set."""
    p, added = Path(path), []
    if not p.exists():
        return added
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            added.append(key)
    return added


def load_config(path: str | Path, datasets=None, models=None) -> dict:
    load_env()                                    # API keys from ./.env (repo root)
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["_path"] = str(path)
    cfg["_datasets"] = list(datasets or cfg["datasets"])
    ids = [m["model"] for m in cfg["models"]]
    cfg["_models"] = list(models or ids)
    unknown = [m for m in cfg["_models"] if m not in ids]
    if unknown:
        raise SystemExit(f"{path}: unknown model id(s) {unknown}; `models` has {ids}")
    return cfg


def results_dir(cfg: dict, ds: str | None = None) -> Path:
    """<results>/<ds>/<config name>/, or <results>/all/<config name>/ when ds is None."""
    return Path(cfg["paths"]["results"]) / (ds or "all") / Path(cfg["_path"]).stem


def model_spec(cfg: dict, model_id: str) -> dict:
    """The `models:` entry of a model under test, by its model id."""
    for m in cfg["models"]:
        if m["model"] == model_id:
            return m
    raise SystemExit(f"{cfg['_path']}: no model {model_id!r} in `models`")


def arg_parser(desc: str) -> argparse.ArgumentParser:
    """Standard options for data/ and experiments/scripts/."""
    ap = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="configs/default.yaml")
    ap.add_argument("-d", "--dataset", action="append", help="restrict to dataset(s)")
    ap.add_argument("-m", "--model", action="append", help="restrict to model key(s)")
    ap.add_argument("--force", action="store_true", help="recompute cached protocol results")
    return ap


def from_args(a: argparse.Namespace) -> dict:
    return load_config(a.config, a.dataset, a.model)


def cells(cfg: dict) -> Iterator[tuple[str, str]]:
    """The selected (dataset, model) pairs."""
    for ds in cfg["_datasets"]:
        for mk in cfg["_models"]:
            yield ds, mk


def check_config(cfg: dict) -> None:
    """Refuse to spend on a config with unfilled `<...>` model ids in the selected cells."""
    specs = {"generator": cfg["generator"], "judge": cfg["judge"]}
    for mk in cfg["_models"]:
        specs[f"model {mk}"] = model_spec(cfg, mk)
    if cfg.get("judge_sensitivity_dataset") in cfg["_datasets"] and cfg.get("alt_judge"):
        specs["alt_judge"] = cfg["alt_judge"]
    bad = [k for k, s in specs.items() if any("<" in str(v) for v in s.values())]
    if bad:
        raise SystemExit(f"{cfg['_path']}: fill in placeholder model ids for {', '.join(bad)}")


def _git() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "note": "not a git repository"}


def record_run(cfg: dict, command: str) -> None:
    """Append seed, environment, git commit and the full config to <results>/all/<config>/run_log.jsonl."""
    pkgs = {}
    for name in ("confledger", "numpy", "scipy", "openai", "anthropic", "datasets"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    entry = {"time": dt.datetime.now().isoformat(timespec="seconds"), "command": command,
             "config_path": cfg["_path"], "seed": cfg["seed"],
             "cells": [f"{ds}__{mk}" for ds, mk in cells(cfg)],
             "env": {"python": platform.python_version(), "platform": platform.platform(),
                     "LLM_BACKEND": os.environ.get("LLM_BACKEND"), "packages": pkgs},
             "git": _git(), "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}
    out = results_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run_log.jsonl").open("a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
