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

from ..answer_parsing import answer_token_prob, p_yes_trailer
from .request import Rejected, Request


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


def sampling_kwargs(req: Request, reasoning_effort: str | None = None,
                    reasoning_tokens: int = 4000) -> dict:
    """Request parameters for a chat completion. Standard models get the request's
    temperature and max_tokens. Reasoning models (gpt-5, o-series; a spec with
    `reasoning_effort`) reject both: they sample at their fixed default temperature and
    take max_completion_tokens, which also covers hidden reasoning, so the output
    budget gets `reasoning_tokens` of headroom."""
    if reasoning_effort:
        return {"max_completion_tokens": req.max_tokens + reasoning_tokens,
                "reasoning_effort": reasoning_effort}
    return {"max_tokens": req.max_tokens, "temperature": req.temperature}


class OpenAICompatBackend:
    def __init__(self, model: str, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", name: str | None = None,
                 reasoning_effort: str | None = None, reasoning_tokens: int = 4000,
                 extra: dict | None = None):
        import openai  # noqa: PLC0415
        self.client = openai.OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env))
        self.model = model
        self.name = name or f"openai_compat:{model}"
        self.reasoning = (reasoning_effort, reasoning_tokens)
        self.extra = dict(extra or {})   # provider-specific request fields, sent as-is (e.g. Gemini thinking off)

    def generate(self, req: Request) -> str:
        kw = {**sampling_kwargs(req, *self.reasoning), **self.extra}
        lp = req.logprobs and not self.reasoning[0]          # reasoning models give no logprobs
        if lp:
            kw.update(logprobs=True, top_logprobs=5)
        r = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": req.prompt}], **kw)
        text = r.choices[0].message.content or ""
        if lp and r.choices[0].logprobs is not None:
            text += p_yes_trailer(answer_token_prob(r.choices[0].logprobs.content))
        return text


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
    rebuild the model once and retry. HTTP 400 / 403 raise `Rejected` (no retry, as
    AgenticSys_v2's FirewallRejection). Other errors propagate to CachedClient's retry loop.
    """

    CALL_TIMEOUT_S = float(os.environ.get("SAFECHAIN_CALL_TIMEOUT_S", "180"))
    STALL_RETRY_S = float(os.environ.get("SAFECHAIN_STALL_RETRY_S", "40"))

    def __init__(self, model: str, name: str | None = None,
                 reasoning_effort: str | None = None, reasoning_tokens: int = 4000):
        from langchain_core.messages import HumanMessage  # noqa: PLC0415
        from langchain_core.prompts import MessagesPlaceholder  # noqa: PLC0415
        from safechain.core.model import amodel  # noqa: PLC0415
        from safechain.prompts import ValidChatPromptTemplate  # noqa: PLC0415
        self._amodel, self._human = amodel, HumanMessage
        self._template = ValidChatPromptTemplate.from_messages([MessagesPlaceholder("messages")])
        self.model = model
        self.name = name or f"safechain:{model}"
        self.reasoning = (reasoning_effort, reasoning_tokens)
        self._llm = None
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True,
                         name="safechain-loop").start()

    async def _build(self):
        self._llm = await asyncio.wait_for(self._amodel(self.model), self.CALL_TIMEOUT_S)

    async def _call(self, req: Request):
        if self._llm is None:
            await self._build()
        kw = sampling_kwargs(req, *self.reasoning)
        if req.logprobs and not self.reasoning[0]:
            kw.update(logprobs=True, top_logprobs=5)
        chain = self._template | self._llm.bind(**kw)
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
            es = str(e)
            if "401" not in es:
                if "403" in es:                 # AgenticSys_v2: FirewallRejection("403")
                    raise Rejected(f"safechain blocked: {es[:300]}") from e
                if "400" in es:
                    raise Rejected(f"safechain bad request: {es[:300]}") from e
                raise
            await self._build()
            reply = await self._call(req)
        c = reply.content
        if isinstance(c, list):
            c = "".join(p if isinstance(p, str) else p.get("text", "") for p in c)
        c = c or ""
        if req.logprobs and not self.reasoning[0]:
            lp = (getattr(reply, "response_metadata", {}) or {}).get("logprobs") or {}
            c += p_yes_trailer(answer_token_prob(lp.get("content")))
        return c

    def generate(self, req: Request) -> str:
        return asyncio.run_coroutine_threadsafe(self._generate(req), self._loop).result()


def make_backend(spec: dict):
    kind = spec["provider"]
    if (kind == "openai_compat" and not spec.get("base_url")      # explicit endpoints stay put
            and os.environ.get("LLM_BACKEND") == "safechain"):
        kind = "safechain"
    reasoning = {k: spec[k] for k in ("reasoning_effort", "reasoning_tokens") if k in spec}
    if kind == "safechain":
        return SafeChainBackend(spec["model"], spec.get("name"), **reasoning)
    if kind == "anthropic":
        return AnthropicBackend(spec["model"], spec.get("name"))
    if kind == "openai_compat":
        return OpenAICompatBackend(spec["model"], spec.get("base_url"),
                                   spec.get("api_key_env", "OPENAI_API_KEY"), spec.get("name"), **reasoning,
                                   extra=spec.get("extra"))
    if kind == "simulated":
        from .simulated_model import SimulatedBackend  # noqa: PLC0415
        return SimulatedBackend(**{k: v for k, v in spec.items() if k != "provider"})
    raise ValueError(f"unknown provider {kind!r}")
