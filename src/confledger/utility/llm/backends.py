"""Real API backends (lazy imports; install the matching optional extra).

- anthropic:   Claude models via the Anthropic SDK (ANTHROPIC_API_KEY)
- openai_compat: any OpenAI-compatible endpoint -- OpenAI, hosted open-weights
                 (Together, Fireworks, vLLM), or an internal enterprise gateway,
                 via base_url + api_key_env.
- safechain:   OpenAI models behind the private SafeChain pipeline (see below).
- simulated:   synthetic model for tests and the dry run (no network).

Transport switch (same convention as AgenticSys_v2): LLM_BACKEND=safechain routes
every `openai_compat` entry without a `base_url` through SafeChain, so one config serves both the dev
machine (direct OpenAI API) and the private environment. The backend name differs
("safechain:<model>"), so the response cache never mixes the two transports.

Sampling with temperature>0 on a provider without a seed parameter is fine: the
cache key includes the sample index, so draws are distinct and reproducible
from the cache.
"""
from __future__ import annotations

import asyncio
import os
import threading

from .request import Request


class AnthropicBackend:
    def __init__(self, model: str, name: str | None = None):
        import anthropic  # noqa: PLC0415
        self.client = anthropic.Anthropic()
        self.model = model
        self.name = name or f"anthropic:{model}"

    def generate(self, req: Request) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=req.max_tokens, temperature=req.temperature,
            messages=[{"role": "user", "content": req.prompt}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAICompatBackend:
    def __init__(self, model: str, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", name: str | None = None):
        import openai  # noqa: PLC0415
        self.client = openai.OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env))
        self.model = model
        self.name = name or f"openai_compat:{model}"

    def generate(self, req: Request) -> str:
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=req.max_tokens, temperature=req.temperature,
            messages=[{"role": "user", "content": req.prompt}])
        return r.choices[0].message.content or ""


class SafeChainBackend:
    """gpt-* models via SafeChain (private env only; mirrors AgenticSys_v2
    llm/safechain_client.py). `await amodel(id)` returns a LangChain
    AzureChatOpenAI subclass; the call goes through ValidChatPromptTemplate for
    compliance, with temperature/max_tokens forwarded by `.bind()`.

    SafeChain is async-only, while the Collector calls `generate` from worker
    threads, so one private event loop runs on a daemon thread and every call is
    submitted to it. Timeouts follow AgenticSys_v2's measurements: stalled calls
    hang rather than run slow, so a first attempt still running at 40s is
    abandoned and re-issued with the full 180s budget. HTTP 401 = token expiry:
    rebuild the model once and retry. Other errors propagate to CachedClient's
    retry loop.
    """

    CALL_TIMEOUT_S = float(os.environ.get("SAFECHAIN_CALL_TIMEOUT_S", "180"))
    STALL_RETRY_S = float(os.environ.get("SAFECHAIN_STALL_RETRY_S", "40"))

    def __init__(self, model: str, name: str | None = None):
        from langchain_core.messages import HumanMessage  # noqa: PLC0415
        from langchain_core.prompts import MessagesPlaceholder  # noqa: PLC0415
        from safechain.core.model import amodel  # noqa: PLC0415
        from safechain.prompts import ValidChatPromptTemplate  # noqa: PLC0415
        self._amodel, self._human = amodel, HumanMessage
        self._template = ValidChatPromptTemplate.from_messages([MessagesPlaceholder("messages")])
        self.model = model
        self.name = name or f"safechain:{model}"
        self._llm = None
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True,
                         name="safechain-loop").start()

    async def _build(self):
        self._llm = await asyncio.wait_for(self._amodel(self.model), self.CALL_TIMEOUT_S)

    async def _call(self, req: Request):
        if self._llm is None:
            await self._build()
        chain = self._template | self._llm.bind(temperature=req.temperature,
                                                 max_tokens=req.max_tokens)
        inp = {"messages": [self._human(content=req.prompt)]}
        first = min(self.STALL_RETRY_S, self.CALL_TIMEOUT_S)
        if first > 0:
            try:
                return await asyncio.wait_for(chain.ainvoke(inp), first)
            except asyncio.TimeoutError:
                pass                    # wedged request: re-issue with the full budget
        return await asyncio.wait_for(chain.ainvoke(inp), self.CALL_TIMEOUT_S)

    async def _generate(self, req: Request) -> str:
        try:
            reply = await self._call(req)
        except Exception as e:
            if "401" not in str(e):
                raise
            await self._build()
            reply = await self._call(req)
        c = reply.content
        if isinstance(c, list):
            c = "".join(p if isinstance(p, str) else p.get("text", "") for p in c)
        return c or ""

    def generate(self, req: Request) -> str:
        return asyncio.run_coroutine_threadsafe(self._generate(req), self._loop).result()


def make_backend(spec: dict):
    kind = spec["provider"]
    if (kind == "openai_compat" and not spec.get("base_url")      # explicit endpoints stay put
            and os.environ.get("LLM_BACKEND") == "safechain"):
        kind = "safechain"
    if kind == "safechain":
        return SafeChainBackend(spec["model"], spec.get("name"))
    if kind == "anthropic":
        return AnthropicBackend(spec["model"], spec.get("name"))
    if kind == "openai_compat":
        return OpenAICompatBackend(spec["model"], spec.get("base_url"),
                                   spec.get("api_key_env", "OPENAI_API_KEY"), spec.get("name"))
    if kind == "simulated":
        from .simulated_model import SimulatedBackend  # noqa: PLC0415
        return SimulatedBackend(**{k: v for k, v in spec.items() if k != "provider"})
    raise ValueError(f"unknown provider {kind!r}")
