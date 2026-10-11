"""Backend selection, the SafeChain backend (against a fake safechain module),
the config placeholder guard, and the run log."""
import asyncio
import os
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

from confledger.utility.config import check_config, load_config, load_env, record_run  # noqa: E402
from confledger.utility.llm import Request, make_backend  # noqa: E402
from confledger.utility.llm.backends import SafeChainBackend  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    """Stands in for SafeAzureChatOpenAI: records bound kwargs and the prompt."""

    def __init__(self, log, fail_401=False, stall_s=0.0):
        self.log, self.fail_401, self.stall_s = log, fail_401, stall_s

    def bind(self, **kw):
        async def call(pv):
            self.log["calls"] += 1
            if self.fail_401:
                raise RuntimeError("Error code: 401 - token expired")
            if self.stall_s and self.log["calls"] == 1:
                await asyncio.sleep(self.stall_s)
            self.log["kw"], self.log["text"] = kw, pv.to_messages()[-1].content
            return AIMessage(content="Answer: Yes")
        return RunnableLambda(lambda pv: None, afunc=call)


@pytest.fixture
def fake_safechain(monkeypatch):
    log = {"builds": 0, "calls": 0, "models": []}

    async def amodel(model_id):
        log["builds"] += 1
        return log["models"].pop(0) if log["models"] else FakeModel(log)

    mods = {"safechain": types.ModuleType("safechain"),
            "safechain.core": types.ModuleType("safechain.core"),
            "safechain.core.model": types.SimpleNamespace(amodel=amodel),
            "safechain.prompts": types.SimpleNamespace(
                ValidChatPromptTemplate=type("ValidChatPromptTemplate", (ChatPromptTemplate,), {}))}
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    return log


def test_env_switch_routes_openai_compat_to_safechain(fake_safechain, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "safechain")
    b = make_backend({"provider": "openai_compat", "model": "gpt-4.1"})
    assert isinstance(b, SafeChainBackend) and b.name == "safechain:gpt-4.1"
    assert b.generate(Request("Is it?", 0.7, max_tokens=300)) == "Answer: Yes"
    assert fake_safechain["kw"] == {"temperature": 0.7, "max_tokens": 300}
    assert fake_safechain["text"] == "Is it?"


def test_explicit_endpoint_not_rerouted(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "safechain")
    monkeypatch.setenv("OPEN_API_KEY", "x")
    b = make_backend({"provider": "openai_compat", "model": "m", "base_url": "https://h/v1",
                      "api_key_env": "OPEN_API_KEY"})
    assert b.name == "openai_compat:m"


def test_401_rebuilds_model_once(fake_safechain):
    fake_safechain["models"] = [FakeModel(fake_safechain, fail_401=True)]
    b = make_backend({"provider": "safechain", "model": "gpt-4.1"})
    assert b.generate(Request("q")) == "Answer: Yes"
    assert fake_safechain["builds"] == 2


def test_stalled_call_is_reissued(fake_safechain, monkeypatch):
    monkeypatch.setattr(SafeChainBackend, "STALL_RETRY_S", 0.05)
    fake_safechain["models"] = [FakeModel(fake_safechain, stall_s=5)]
    b = make_backend({"provider": "safechain", "model": "gpt-4.1"})
    assert b.generate(Request("q")) == "Answer: Yes"
    assert fake_safechain["calls"] == 2


def test_check_config_rejects_placeholders():
    cfg = load_config(ROOT / "configs/smoke.yaml")
    cfg["generator"] = {"provider": "openai_compat", "model": "<generator-model-id>"}
    with pytest.raises(SystemExit, match="generator"):
        check_config(cfg)
    check_config(load_config(ROOT / "configs/simulation.yaml"))



def test_record_run_logs_seed_env_git(tmp_path):
    cfg = load_config(ROOT / "configs/simulation.yaml")
    cfg["paths"] = {**cfg["paths"], "results": str(tmp_path)}
    record_run(cfg, "collect")
    e = json.loads((tmp_path / "all" / "simulation" / "run_log.jsonl").read_text().splitlines()[-1])
    assert e["seed"] == cfg["seed"] and e["command"] == "collect"
    assert {"python", "packages", "LLM_BACKEND"} <= set(e["env"]) and "commit" in e["git"]
    assert e["config"]["split"] == cfg["split"]


def test_load_env_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# keys\nexport CONFLEDGER_T1="abc"\nCONFLEDGER_T2=xyz\n')
    monkeypatch.delenv("CONFLEDGER_T1", raising=False)
    monkeypatch.setenv("CONFLEDGER_T2", "from-shell")
    assert load_env(env) == ["CONFLEDGER_T1"]
    assert os.environ["CONFLEDGER_T1"] == "abc" and os.environ["CONFLEDGER_T2"] == "from-shell"


def test_reasoning_model_parameters(fake_safechain, monkeypatch):
    # standard model: temperature + max_tokens; reasoning model (gpt-5): neither
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    seen = {}
    fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
        create=lambda **kw: seen.update(kw) or types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]))))
    b = make_backend({"provider": "openai_compat", "model": "gpt-4.1"})
    b.client = fake
    b.generate(Request("q", 0.7, max_tokens=300))
    assert seen["temperature"] == 0.7 and seen["max_tokens"] == 300
    seen.clear()
    b = make_backend({"provider": "openai_compat", "model": "gpt-5", "reasoning_effort": "low"})
    b.client = fake
    b.generate(Request("q", 0.7, max_tokens=60))
    assert "temperature" not in seen and "max_tokens" not in seen
    assert seen["max_completion_tokens"] == 60 + 4000 and seen["reasoning_effort"] == "low"
    # the same rule on the SafeChain transport
    monkeypatch.setenv("LLM_BACKEND", "safechain")
    b = make_backend({"provider": "openai_compat", "model": "gpt-5", "reasoning_effort": "minimal",
                      "reasoning_tokens": 1000})
    assert b.generate(Request("q", 0.7, max_tokens=60)) == "Answer: Yes"
    assert fake_safechain["kw"] == {"max_completion_tokens": 1060, "reasoning_effort": "minimal"}
