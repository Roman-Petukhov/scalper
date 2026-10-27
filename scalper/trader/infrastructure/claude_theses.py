"""Аналитик тезисов на Claude: веб-поиск и чтение источников, затем строгий JSON по схеме domain/theses.py.

Два запроса на один отчёт: исследование (web_search / web_fetch, текст со ссылками; пауза сервера pause_turn —
продолжаем тем же диалогом) и оформление (без инструментов, output_config.format по JSON-схеме — с цитатами
структурный вывод не совместим, поэтому ссылки переносятся в поле sources). Еженедельная сверка — один запрос с
поиском и схемой поменьше: что изменилось по катализаторам и условиям отмены за неделю.

Отказ модели по политике (stop_reason refusal) перезапускается на сервере другой моделью (fallbacks "default")."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import anthropic

log = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
WEB_TOOLS = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 40},
             {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 20}]
MAX_PAUSES = 8
PRICE_IN, PRICE_OUT, PRICE_SEARCH = 4.0 / 1e6, 20.0 / 1e6, 10.0 / 1000      # Opus 5.5, $ за токен и за поиск

RESEARCH_SYSTEM = """Ты — аналитик тематического криптофонда. Задача: тезисы на 1–3 года о том, куда могут пойти деньги
в крипте из-за мировых сдвигов (технологии, регулирование, макроэкономика, поведение институционалов).

Правила:
- Ищи в интернете свежие первоисточники: регуляторы, отчёты исследовательских команд, данные DefiLlama / rwa.xyz /
  Dune, крупные издания. Каждое утверждение с цифрой или датой — со ссылкой. Нет источника — не пиши.
- У каждого тезиса: сдвиг в мире → причинная цепочка до крипты → кто реально получает деньги (держатель токена или
  компания без токена) → опережающие индикаторы из каталога → катализаторы с датами → условие отмены.
- Отделяй силу сдвига от того, доходит ли выручка до держателя токена, и от того, учла ли это уже цена.
- Включай риск-тезисы (что может ударить по рынку), а не только идеи для покупки.
- Учитывай макро-фон и режим рынка: тезисы — про накопление, не про покупку на хайпе.
- Если есть прошлая версия тезисов, обнови её: оставь, усиль, ослабь или сними каждый тезис и объясни почему.
- Пиши по-русски, коротко и конкретно, без рекламных слов. Цифры из разных источников расходятся — давай диапазоны."""

FORMAT_SYSTEM = """Оформи исследование в JSON по схеме. Только то, что есть в тексте исследования, ничего не добавляй.
Индикаторы — только id из каталога. В sources — ссылки, на которые опирается тезис (из исследования).
Тикеры кандидатов — как на биржах (HYPE, LINK). Если у сдвига нет токена-получателя, оставь список кандидатов пустым."""

CHECK_SYSTEM = """Ты следишь за тезисами тематического криптофонда. Найди в интернете новости за последние 7 дней,
которые касаются катализаторов и условий отмены каждого тезиса. Отметь только существенное, со ссылками.
Пиши по-русски, коротко. Ничего не выдумывай: нет новостей — так и напиши."""

CHECK_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "change", "note", "sources"],
        "properties": {
            "id": {"type": "string"},
            "change": {"type": "string", "enum": ["none", "support", "against", "kill"]},
            "note": {"type": "string"},
            "sources": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["title", "url"],
                "properties": {"title": {"type": "string"}, "url": {"type": "string"}}}},
        }}}},
}


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0

    def __add__(self, o: Usage) -> Usage:
        return Usage(self.input_tokens + o.input_tokens, self.output_tokens + o.output_tokens,
                     self.searches + o.searches)

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * PRICE_IN + self.output_tokens * PRICE_OUT + self.searches * PRICE_SEARCH


def _usage(msg) -> Usage:
    u = getattr(msg, "usage", None)
    if u is None:
        return Usage()
    stu = getattr(u, "server_tool_use", None)
    searches = int(getattr(stu, "web_search_requests", 0) or 0) if stu is not None else 0
    inp = int(u.input_tokens or 0) + int(getattr(u, "cache_read_input_tokens", 0) or 0) \
        + int(getattr(u, "cache_creation_input_tokens", 0) or 0)
    return Usage(inp, int(u.output_tokens or 0), searches)


def _text(msg) -> str:
    return "\n".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _sources(msg) -> list[str]:
    """Ссылки из цитат ответа и результатов поиска — чтобы оформление не потеряло источники."""
    urls: list[str] = []
    for b in msg.content:
        for c in getattr(b, "citations", None) or []:
            u = getattr(c, "url", None)
            if u and u not in urls:
                urls.append(u)
        if getattr(b, "type", "") == "web_search_tool_result" and isinstance(getattr(b, "content", None), list):
            for r in b.content:
                u = getattr(r, "url", None)
                if u and u not in urls:
                    urls.append(u)
    return urls


class RefusedError(RuntimeError):
    pass


class ClaudeThesesWriter:
    def __init__(self, api_key: str, model: str = MODEL, client: anthropic.AsyncAnthropic | None = None) -> None:
        self.model = model
        self.client = client or anthropic.AsyncAnthropic(api_key=api_key, max_retries=3, timeout=900.0)

    async def _call(self, **kw):
        async with self.client.beta.messages.stream(model=self.model, betas=[FALLBACK_BETA], fallbacks="default",
                                                    **kw) as stream:
            msg = await stream.get_final_message()
        if msg.stop_reason == "refusal":
            raise RefusedError(f"модель отказалась: {getattr(msg.stop_details, 'category', None)}")
        return msg

    async def _research(self, system: str, prompt: str, effort: str) -> tuple[str, list[str], Usage]:
        messages: list = [{"role": "user", "content": prompt}]
        usage, texts, urls = Usage(), [], []
        for _ in range(MAX_PAUSES + 1):
            msg = await self._call(max_tokens=64000, system=system, messages=messages, tools=WEB_TOOLS,
                                   output_config={"effort": effort})
            usage = usage + _usage(msg)
            texts.append(_text(msg))
            urls += [u for u in _sources(msg) if u not in urls]
            if msg.stop_reason != "pause_turn":
                break
            messages.append({"role": "assistant", "content": msg.content})
        else:
            log.warning("тезисы: исследование не закончилось за %d пауз", MAX_PAUSES)
        return "\n".join(t for t in texts if t), urls, usage

    async def _format(self, system: str, text: str, schema: dict) -> tuple[dict, Usage]:
        msg = await self._call(max_tokens=32000, system=system, messages=[{"role": "user", "content": text}],
                               output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
        if msg.stop_reason == "max_tokens":
            raise RuntimeError("оформление тезисов обрезано по max_tokens")
        return json.loads(_text(msg)), _usage(msg)

    async def write(self, context: str, schema: dict) -> tuple[dict, Usage]:
        """Полный пересчёт: исследование с поиском, затем JSON по схеме."""
        research, urls, u1 = await self._research(RESEARCH_SYSTEM, context, "high")
        if not research.strip():
            raise RuntimeError("исследование вернулось пустым")
        body = (f"Исследование:\n{research}\n\nНайденные ссылки:\n" + "\n".join(urls[:200]) +
                f"\n\nКонтекст (каталог индикаторов и скринер):\n{context}")
        data, u2 = await self._format(FORMAT_SYSTEM, body, schema)
        return data, u1 + u2

    async def check(self, context: str) -> tuple[dict, Usage]:
        """Недельная сверка: новости по катализаторам и условиям отмены текущих тезисов."""
        research, urls, u1 = await self._research(CHECK_SYSTEM, context, "low")
        body = f"Сверка:\n{research}\n\nНайденные ссылки:\n" + "\n".join(urls[:100]) + f"\n\nТезисы:\n{context}"
        data, u2 = await self._format("Оформи сверку в JSON по схеме. id — как у тезисов.", body, CHECK_SCHEMA)
        return data, u1 + u2

    async def aclose(self) -> None:
        await self.client.close()


