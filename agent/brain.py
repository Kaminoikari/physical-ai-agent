"""L3 大腦：呼叫 LLM 把指令拆解成結構化 Plan。

LLMClient 是可抽換介面：真實版 AnthropicClient 打 API，測試版 FakeClient 回罐頭
JSON（零 API 費，對應「外部依賴一律 mock」）。parse_plan 負責把 LLM 回傳的 JSON
防呆地轉成 Plan。
"""

from __future__ import annotations

import json
import re
import time
from typing import Protocol

from agent.prompts import SYSTEM_PROMPT
from agent.schemas import Plan, Step

_REQUIRED_FIELDS = ("reasoning", "in_scope", "needs_clarification", "clarification_question", "plan")


class ParseError(ValueError):
    """LLM 回傳無法解析成合法 Plan。"""


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class FakeClient:
    """測試用：回傳預先準備好的回應，不打 API。

    傳入 str 則每次都回同一個；傳入 list[str] 則依序回傳（用完停在最後一個），
    可模擬「觀察後再規劃」的多輪對話。
    """

    def __init__(self, response: str | list[str]) -> None:
        self._sequence = response if isinstance(response, list) else None
        self._single = None if isinstance(response, list) else response
        self._index = 0

    def complete(self, system: str, user: str) -> str:
        if self._sequence is not None:
            reply = self._sequence[min(self._index, len(self._sequence) - 1)]
            self._index += 1
            return reply
        return self._single


class AnthropicClient:
    """真實版：打 Anthropic API。API key 由 os.environ 讀取（透過 anthropic SDK）。"""

    def __init__(self, model: str = "claude-sonnet-4-6", max_retries: int = 2) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_retries = max_retries

    def complete(self, system: str, user: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return resp.content[0].text
            except (self._anthropic.APIStatusError, self._anthropic.APIConnectionError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Anthropic API 連續失敗：{last_exc}") from last_exc


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return fence.group(1).strip() if fence else text


def parse_plan(raw: str) -> Plan:
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise ParseError(f"LLM 回傳不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ParseError("LLM 回傳不是 JSON 物件")
    missing = [f for f in _REQUIRED_FIELDS if f not in data]
    if missing:
        raise ParseError(f"LLM 回傳缺少欄位：{missing}")
    try:
        steps = [Step(skill=s["skill"], arg=s.get("arg", "")) for s in data["plan"]]
    except (TypeError, KeyError) as exc:
        raise ParseError(f"plan 步驟格式錯誤：{exc}") from exc
    return Plan(
        reasoning=str(data["reasoning"]),
        in_scope=bool(data["in_scope"]),
        needs_clarification=bool(data["needs_clarification"]),
        clarification_question=str(data["clarification_question"]),
        steps=steps,
    )


class Brain:
    def __init__(self, client: LLMClient, system_prompt: str = SYSTEM_PROMPT) -> None:
        self._client = client
        self._system_prompt = system_prompt

    def decompose(self, instruction: str, observation: str | None = None) -> Plan:
        user = instruction if observation is None else f"{instruction}\n\n現場觀察：{observation}"
        raw = self._client.complete(self._system_prompt, user)
        return parse_plan(raw)
