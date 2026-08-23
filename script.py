#!/usr/bin/env python3
"""
Генерация Telegram-бота (aiogram 3.x) по текстовому ТЗ через цепочку LangChain.

Зависимости:
    pip install -r requirements.txt

Переменные окружения (файл .env, см. .env.example):
    PROXY_API_KEY     — ключ ProxyAPI (обязателен; также принимается OPENAI_API_KEY)
    PROXY_API_BASE_URL — URL ProxyAPI (по умолчанию https://api.proxyapi.ru/openai/v1)
    PROXY_API_MODEL   — модель (по умолчанию gpt-4o-mini; также OPENAI_MODEL)
    PROXY_API_TIMEOUT — таймаут одного запроса в секундах (по умолчанию 180)
    OPENAI_MAX_TOKENS — лимит токенов ответа для code/review (по умолчанию 4096)

Пример:
    python script.py "Бот, который отправляет случайные мемы"
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from openai import APIStatusError, APITimeoutError, OpenAI

load_dotenv()

logger = logging.getLogger("botgen")

# Endpoint OpenAI через ProxyAPI (официальный SDK openai совместим с этим URL)
DEFAULT_PROXYAPI_BASE_URL = "https://api.proxyapi.ru/openai/v1"

_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
}


class OpenAIChat(Runnable):
    """Обёртка openai.OpenAI (через ProxyAPI) для LCEL-цепочек LangChain."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        temperature: float = 0.2,
        base_url: str = DEFAULT_PROXYAPI_BASE_URL,
        timeout: float = 180.0,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.timeout = timeout
        self.max_tokens = max_tokens or int(os.getenv("OPENAI_MAX_TOKENS", "4096"))

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> str:
        messages = self._to_openai_messages(input)
        max_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        logger.info(
            "ProxyAPI-запрос: base_url=%s model=%s messages=%s max_tokens=%s timeout=%ss",
            self.base_url,
            self.model,
            len(messages),
            max_tokens,
            int(self.timeout),
        )
        logger.info(
            "ProxyAPI: ожидание ответа (модель %s может отвечать 30–120 с на шаг)...",
            self.model,
        )

        stop = threading.Event()
        started = time.perf_counter()

        def heartbeat() -> None:
            while not stop.wait(10):
                elapsed = int(time.perf_counter() - started)
                logger.info("ProxyAPI: всё ещё ждём ответ... %s с", elapsed)

        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens,
                messages=messages,
            )
        except APITimeoutError as exc:
            raise RuntimeError(
                f"ProxyAPI не ответил за {int(self.timeout)} с. "
                "Попробуйте PROXY_API_MODEL=gpt-4o-mini или увеличьте PROXY_API_TIMEOUT в .env."
            ) from exc
        except APIStatusError as exc:
            if exc.status_code == 400 and "model not supported" in str(exc).lower():
                raise RuntimeError(
                    f"ProxyAPI не поддерживает модель «{self.model}». "
                    "Укажите в .env одну из моделей ProxyAPI, например: "
                    "gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-3.5-turbo "
                    "(см. https://proxyapi.ru/docs/openai-models). "
                    f"Ответ API: {exc.message}"
                ) from exc
            raise
        finally:
            stop.set()
            hb.join(timeout=1)

        elapsed = time.perf_counter() - started
        content = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        if usage is not None:
            logger.info(
                "ProxyAPI-ответ за %.1f с: finish=%s prompt_tokens=%s completion_tokens=%s",
                elapsed,
                getattr(response.choices[0], "finish_reason", None),
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None),
            )
        else:
            logger.info(
                "ProxyAPI-ответ за %.1f с: finish=%s",
                elapsed,
                getattr(response.choices[0], "finish_reason", None),
            )
        if not content:
            logger.error("ProxyAPI вернул пустой ответ")
            raise RuntimeError("ProxyAPI вернул пустой ответ.")
        logger.debug("Фрагмент ответа LLM: %s", content[:300].replace("\n", " "))
        return content

    @staticmethod
    def _to_openai_messages(prompt_value: Any) -> list[dict[str, str]]:
        if hasattr(prompt_value, "to_messages"):
            raw_messages = prompt_value.to_messages()
        elif isinstance(prompt_value, list):
            raw_messages = prompt_value
        else:
            return [{"role": "user", "content": str(prompt_value)}]

        converted: list[dict[str, str]] = []
        for message in raw_messages:
            if isinstance(message, BaseMessage):
                role = _ROLE_MAP.get(message.type, "user")
                converted.append({"role": role, "content": str(message.content)})
            elif isinstance(message, dict):
                converted.append(
                    {
                        "role": str(message.get("role", "user")),
                        "content": str(message.get("content", "")),
                    }
                )
            else:
                converted.append({"role": "user", "content": str(message)})
        return converted


def resolve_proxyapi_key() -> str:
    api_key = os.getenv("PROXY_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Не задан ключ ProxyAPI. Укажите PROXY_API_KEY в .env "
            "(или OPENAI_API_KEY). См. .env.example и https://proxyapi.ru/"
        )
    return api_key


def resolve_proxyapi_base_url() -> str:
    return (
        os.getenv("PROXY_API_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or DEFAULT_PROXYAPI_BASE_URL
    )


def normalize_proxyapi_model(raw_model: str) -> str:
    """Приводит имя модели к формату ProxyAPI (ASCII, нижний регистр)."""
    model = raw_model.strip()
    # Частая ошибка: «умные» дефисы из копипаста (GPT‑5.3‑Codex)
    for ch in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        model = model.replace(ch, "-")
    return model.lower()


def resolve_proxyapi_model() -> str:
    raw = os.getenv("PROXY_API_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    return normalize_proxyapi_model(raw)


def resolve_proxyapi_timeout() -> float:
    return float(os.getenv("PROXY_API_TIMEOUT", "180"))


def build_llm(max_tokens: int | None = None) -> OpenAIChat:
    """Клиент OpenAI SDK, направленный на ProxyAPI (base_url обязателен)."""
    api_key = resolve_proxyapi_key()
    base_url = resolve_proxyapi_base_url()
    model = resolve_proxyapi_model()
    timeout = resolve_proxyapi_timeout()

    logger.info(
        "LLM через ProxyAPI: base_url=%s model=%s timeout=%ss",
        base_url,
        model,
        int(timeout),
    )
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    return OpenAIChat(
        client=client,
        model=model,
        temperature=0.2,
        base_url=base_url,
        timeout=timeout,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Промпты звеньев:
# analysis → tools → structure → code → review
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты аналитик Telegram-ботов. По текстовому ТЗ составь структурированное "
            "техническое задание. Отвечай только структурой, без лишнего текста.",
        ),
        (
            "human",
            "Техническое описание бота:\n{task}\n\n"
            "Сформируй структуру строго в таком виде:\n"
            "КОМАНДЫ:\n"
            "- /start — ...\n"
            "- /help — ...\n"
            "- другие команды по смыслу ТЗ\n\n"
            "ОБРАБОТЧИЧИ:\n"
            "- command: /start → handler: start_handler\n"
            "- message: текст → handler: ...\n"
            "- callback: если нужны inline-кнопки\n\n"
            "ФОРМАТ_ОТВЕТА:\n"
            "- как бот отвечает пользователю (текст, фото, клавиатура и т.д.)\n\n"
            "БАЗА_ДАННЫХ:\n"
            "- да/нет и зачем (если да — какой движок, лучше sqlite)\n\n"
            "ЗАВИСИМОСТИ:\n"
            "- aiogram>=3.4.0\n"
            "- другие пакеты только если реально нужны для ТЗ\n",
        ),
    ]
)

TOOLS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты архитектор Telegram-ботов. По анализу ТЗ подбираешь конкретный "
            "инструментарий для генерации кода. Отвечай только структурой, без лишнего текста.",
        ),
        (
            "human",
            "Исходное ТЗ:\n{task}\n\n"
            "Анализ задания:\n{analysis}\n\n"
            "Подбери инструменты для генерации бота строго в таком виде:\n"
            "ФРЕЙМВОРК:\n"
            "- aiogram 3.x (обязательно)\n\n"
            "БИБЛИОТЕКИ:\n"
            "- пакет == зачем нужен (только реально необходимые)\n\n"
            "ВНЕШНИЕ_API:\n"
            "- имя, публичный URL, что запрашиваем; если API не нужен — «нет»\n\n"
            "ХРАНЕНИЕ:\n"
            "- память / sqlite / файл и почему\n\n"
            "ИНСТРУМЕНТЫ_AIOGRAM:\n"
            "- Router, FSM, фильтры F, inline/reply-клавиатуры, middleware — что именно и зачем\n\n"
            "ЗАПРЕЩЕНО:\n"
            "- заглушки, вымышленные ключи, несуществующие API, синхронные handlers\n",
        ),
    ]
)

STRUCTURE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты tech lead. По анализу и списку инструментов проектируешь структуру "
            "кода Telegram-бота. Отвечай только каркасом, без готовой реализации.",
        ),
        (
            "human",
            "Исходное ТЗ:\n{task}\n\n"
            "Анализ задания:\n{analysis}\n\n"
            "Подобранные инструменты:\n{tools}\n\n"
            "Создай структуру кода одного файла bot.py строго в таком виде:\n"
            "ИМПОРТЫ:\n"
            "- asyncio, logging, os, sys, dotenv.load_dotenv\n"
            "- aiogram: Bot, Dispatcher, F, Router; filters.Command; types.Message\n"
            "- прочие модули по ТЗ (aiohttp, random и т.д.)\n\n"
            "КОНСТАНТЫ_И_КОНФИГ:\n"
            "- load_dotenv(); BOT_TOKEN = os.getenv('BOT_TOKEN'); проверка + sys.exit\n\n"
            "ВСПОМОГАТЕЛЬНЫЕ_ФУНКЦИИ:\n"
            "- имя(аргументы) -> результат — назначение\n\n"
            "ОБРАБОТЧИКИ:\n"
            "- async def имя(message/callback) — команда/событие — что делает\n\n"
            "ТОЧКА_ВХОДА:\n"
            "- async def main() и asyncio.run(main())\n\n"
            "ПОРЯДОК_В_ФАЙЛЕ:\n"
            "- нумерованный список блоков сверху вниз\n"
            "Не пиши полный код, только каркас и сигнатуры.\n",
        ),
    ]
)

CODE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты senior Python-разработчик. Пишешь полный рабочий код Telegram-бота "
            "на aiogram 3.x. Код должен запускаться командой python bot.py без правок. "
            "Отвечай только Python-кодом, без markdown и пояснений.",
        ),
        (
            "human",
            "Исходное ТЗ:\n{task}\n\n"
            "Анализ задания:\n{analysis}\n\n"
            "Подобранные инструменты:\n{tools}\n\n"
            "Структура кода:\n{structure}\n\n"
            "Реализуй ОДИН полный файл bot.py. Начни файл строго с этого каркаса "
            "(дополни его логикой по ТЗ, каркас не удаляй):\n"
            "import asyncio\n"
            "import logging\n"
            "import os\n"
            "import sys\n\n"
            "from aiogram import Bot, Dispatcher, F, Router\n"
            "from aiogram.filters import Command\n"
            "from aiogram.types import Message\n"
            "from dotenv import load_dotenv\n\n"
            "load_dotenv()\n\n"
            "logging.basicConfig(level=logging.INFO)\n"
            "logger = logging.getLogger(__name__)\n\n"
            "BOT_TOKEN = os.getenv('BOT_TOKEN')\n"
            "if not BOT_TOKEN:\n"
            "    sys.exit('BOT_TOKEN не задан. Добавьте его в .env')\n\n"
            "Обязательные правила aiogram 3.x:\n"
            "1. Фильтр текста — F.text; ЗАПРЕЩЕНО: Text из aiogram.filters.\n"
            "2. Все обработчики — async def; регистрация: router.message.register(handler, Filter), "
            "НЕ register(handler()).\n"
            "3. Router подключить: dp.include_router(router).\n"
            "4. Запуск: async def main() + if __name__ == '__main__': asyncio.run(main()).\n"
            "5. В main: await dp.start_polling(bot).\n"
            "6. Каждый используемый модуль импортировать явно (aiohttp, random и т.д.).\n"
            "7. HTTP в async-обработчиках — через aiohttp, не блокируй event loop requests.\n"
            "8. /start и /help обязательны.\n"
            "9. Никаких TODO, pass вместо логики, заглушек.\n"
            "10. load_dotenv() обязателен для чтения BOT_TOKEN из .env.\n"
            "Верни только исходный код файла.\n",
        ),
    ]
)

REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты ревьюер Python-кода. Исправляешь Telegram-бота на aiogram 3.x так, "
            "чтобы python bot.py работал сразу. Верни только полный исправленный код.",
        ),
        (
            "human",
            "Анализ задания:\n{analysis}\n\n"
            "Инструменты:\n{tools}\n\n"
            "Заложенная структура:\n{structure}\n\n"
            "Сгенерированный код:\n{code}\n\n"
            "{syntax_feedback}\n\n"
            "Исправь ВСЕ проблемы из feedback и проверь чеклист:\n"
            "1. Синтаксис Python 3.10+.\n"
            "2. Импорты: os, sys, asyncio, logging, dotenv.load_dotenv — и все используемые модули.\n"
            "3. load_dotenv() вызывается до os.getenv('BOT_TOKEN').\n"
            "4. Нет Text из aiogram.filters — только F.text.\n"
            "5. async def main(), asyncio.run(main()), Bot, Dispatcher, Router, start_polling.\n"
            "6. /start и /help зарегистрированы через Command().\n"
            "7. Код компилируется и запускается без ручных правок.\n"
            "Верни только исходный код файла.\n",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)

# Модули: паттерн использования -> строка импорта
_IMPORT_RULES: list[tuple[str, str, str]] = [
    (r"\bos\.", "os", "import os"),
    (r"\bsys\.", "sys", "import sys"),
    (r"\basyncio\.", "asyncio", "import asyncio"),
    (r"\blogging\.", "logging", "import logging"),
    (r"\brandom\.", "random", "import random"),
    (r"\bjson\.", "json", "import json"),
    (r"\bre\.", "re", "import re"),
    (r"\brequests\.", "requests", "import requests"),
    (r"\baiohttp\.", "aiohttp", "import aiohttp"),
    (r"\bsqlite3\.", "sqlite3", "import sqlite3"),
]

_FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r"from\s+aiogram\.filters\s+import\s+[^#\n]*\bText\b", "Text из aiogram.filters не существует в aiogram 3"),
    (r"\bText\s*\(\s*\)", "фильтр Text() недопустим — используй F.text"),
    (r"(?<![.\w])exit\s*\(", "используй sys.exit(), не exit()"),
    (
        r"\.message\.register\s*\(\s*\w+\s*\(\s*\)",
        "неверная регистрация: register(handler, filter), не register(handler())",
    ),
]


def setup_logging(level: str | None = None) -> None:
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric = getattr(logging, resolved, None)
    invalid_level = not isinstance(numeric, int)
    if invalid_level:
        numeric = logging.INFO
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    if invalid_level:
        logger.warning("Неизвестный LOG_LEVEL=%s, использую INFO", resolved)
    logger.debug("Логгер инициализирован, уровень=%s", logging.getLevelName(numeric))


def extract_python_code(text: str) -> str:
    """Достаёт Python-код из ответа LLM (срезает markdown-ограждения)."""
    text = (text or "").strip()
    if not text:
        logger.error("LLM вернул пустой ответ")
        raise ValueError("LLM вернул пустой ответ.")

    fences = _FENCE_RE.findall(text)
    if fences:
        logger.debug("Из ответа вырезано markdown-ограждение")
        text = max(fences, key=len).strip()

    # На случай преамбулы до первого import / shebang
    for marker in ("#!/usr/bin/env python", "from __future__", "import ", "from "):
        idx = text.find(marker)
        if idx > 0:
            text = text[idx:]
            break
    return text.strip() + "\n"


def syntax_error_message(code: str) -> str | None:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"{exc.msg} (строка {exc.lineno}, колонка {exc.offset})"
    return None


def compile_error_message(code: str) -> str | None:
    try:
        compile(code, "bot.py", "exec")
    except Exception as exc:
        return str(exc)
    return None


def _collect_imported_modules(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                name = alias.asname or alias.name
                if name == "*":
                    continue
                imported.add(name.split(".")[0])
                if node.module:
                    imported.add(node.module.split(".")[0])
    return imported


def _has_command_handler(code: str, command: str) -> bool:
    """Проверяет наличие обработчика команды в любом стиле aiogram 3."""
    patterns = [
        rf"""Command\s*\(\s*['"]{command}['"]\s*\)""",
        rf"""Command\s*\(\s*commands\s*=\s*\[\s*['"]{command}['"]\s*\]\s*\)""",
        rf"""['"]/{command}['"]""",
        rf"async def {command}_handler\b",
    ]
    if command == "start":
        patterns.append(r"\bCommandStart\b")
    return any(re.search(p, code) for p in patterns)


def _has_command_registration(code: str) -> bool:
    return bool(
        re.search(
            r"Command\s*\(|CommandStart|\.message\.register|include_router",
            code,
        )
    )


def validate_bot_code(code: str) -> list[str]:
    """Полная статическая проверка сгенерированного bot.py."""
    issues: list[str] = []

    syntax = syntax_error_message(code)
    if syntax:
        issues.append(f"синтаксис: {syntax}")

    compile_err = compile_error_message(code)
    if compile_err and not syntax:
        issues.append(f"компиляция: {compile_err}")

    for pattern, message in _FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            issues.append(message)

    required_snippets = [
        ("load_dotenv()", "нет вызова load_dotenv()"),
        ("os.getenv", "токен должен читаться через os.getenv"),
        ("BOT_TOKEN", "ожидается переменная BOT_TOKEN"),
        ("async def main", "нет async def main()"),
        ("asyncio.run(main())", "нет asyncio.run(main())"),
        ("Dispatcher", "нет Dispatcher"),
        ("start_polling", "нет dp.start_polling"),
        ("logging.getLogger", "нет logger = logging.getLogger(...)"),
    ]
    for snippet, message in required_snippets:
        if snippet not in code:
            issues.append(message)

    if not _has_command_handler(code, "start"):
        issues.append("нет обработчика /start")
    if not _has_command_handler(code, "help"):
        issues.append("нет обработчика /help")
    if not _has_command_registration(code):
        issues.append("нет регистрации обработчиков")

    if "from aiogram" not in code and "import aiogram" not in code:
        issues.append("нет импорта aiogram")

    imported = _collect_imported_modules(code)
    for pattern, module, import_line in _IMPORT_RULES:
        if re.search(pattern, code) and module not in imported:
            issues.append(f"используется {module}, но нет «{import_line}»")

    if "load_dotenv" in code and "dotenv" not in imported:
        issues.append("используется load_dotenv, но нет from dotenv import load_dotenv")

    if issues:
        logger.warning("Валидация bot.py: %s", "; ".join(issues))
    else:
        logger.info("Валидация bot.py пройдена")
    return issues


def has_valid_startup(code: str) -> list[str]:
    """Обратная совместимость — делегирует validate_bot_code."""
    return validate_bot_code(code)


def _insert_after_imports(code: str, lines_to_insert: list[str]) -> str:
    if not lines_to_insert:
        return code
    src_lines = code.splitlines()
    insert_at = 0
    for idx, line in enumerate(src_lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) or not stripped:
            insert_at = idx + 1
        else:
            break
    merged = src_lines[:insert_at] + lines_to_insert + [""] + src_lines[insert_at:]
    return "\n".join(merged).strip() + "\n"


def _insert_before_main(code: str, block: str) -> str:
    match = re.search(r"^async def main\s*\(", code, flags=re.MULTILINE)
    if not match:
        return code.rstrip() + "\n\n" + block.strip() + "\n"
    idx = match.start()
    return code[:idx].rstrip() + "\n\n" + block.strip() + "\n\n" + code[idx:]


def _insert_before_polling(code: str, lines: list[str]) -> str:
    block = "\n".join(lines)

    def repl(match: re.Match[str]) -> str:
        indent = match.group(1)
        indented = "\n".join(f"{indent}{line.strip()}" for line in lines)
        return f"{indented}\n\n{match.group(0)}"

    updated, count = re.subn(
        r"^(\s*)await dp\.start_polling\(bot\)",
        repl,
        code,
        count=1,
        flags=re.MULTILINE,
    )
    return updated if count else code.replace("await dp.start_polling(bot)", block + "\n\n    await dp.start_polling(bot)", 1)


def _inject_handler_registration(code: str, handler: str, command: str) -> str:
    if re.search(rf"""Command\s*\(\s*['"]{command}['"]\s*\)""", code):
        return code

    reg_router = f'router.message.register({handler}, Command("{command}"))'
    reg_dp = f'dp.message.register({handler}, Command("{command}"))'

    if "router = Router()" in code:
        return code.replace(
            "router = Router()",
            f"router = Router()\n    {reg_router}",
            1,
        )
    if "dp = Dispatcher()" in code and "router = Router()" not in code:
        patched = code.replace(
            "dp = Dispatcher()",
            f"dp = Dispatcher()\n    router = Router()\n    {reg_router}",
            1,
        )
        if "include_router(router)" not in patched:
            patched = _insert_before_polling(patched, ["dp.include_router(router)"])
        return patched
    return _insert_before_polling(code, [reg_dp])


def ensure_required_handlers(code: str) -> str:
    """Гарантирует /start и /help с регистрацией Command — без участия LLM."""
    fixed = code

    if "from aiogram.filters import Command" not in fixed:
        fixed = _insert_after_imports(fixed, ["from aiogram.filters import Command"])
    if "Message" in fixed and "from aiogram.types import Message" not in fixed:
        fixed = _insert_after_imports(fixed, ["from aiogram.types import Message"])
    if "router = Router()" not in fixed and "Router" not in fixed:
        fixed = _insert_after_imports(fixed, ["from aiogram import Bot, Dispatcher, F, Router"])

    if not re.search(r"async def start_handler\b", fixed):
        fixed = _insert_before_main(
            fixed,
            '''
async def start_handler(message: Message) -> None:
    await message.answer("Привет! Бот готов к работе. /help — справка.")
'''.strip(),
        )

    if not re.search(r"async def help_handler\b", fixed):
        fixed = _insert_before_main(
            fixed,
            '''
async def help_handler(message: Message) -> None:
    await message.answer("Команды:\\n/start — начало\\n/help — справка")
'''.strip(),
        )

    if not _has_command_handler(fixed, "start"):
        fixed = _inject_handler_registration(fixed, "start_handler", "start")
    if not _has_command_handler(fixed, "help"):
        fixed = _inject_handler_registration(fixed, "help_handler", "help")

    if "include_router(router)" not in fixed and "router = Router()" in fixed:
        fixed = _insert_before_polling(fixed, ["dp.include_router(router)"])

    return fixed.strip() + "\n"


def auto_fix_bot_code(code: str) -> str:
    """Детерминированные правки типичных ошибок LLM."""
    fixed = code

    # aiogram 2 -> aiogram 3: Text -> F.text
    fixed = re.sub(
        r"from\s+aiogram\.filters\s+import\s+Command\s*,\s*Text",
        "from aiogram.filters import Command",
        fixed,
    )
    fixed = re.sub(
        r"from\s+aiogram\.filters\s+import\s+Text\s*,\s*Command",
        "from aiogram.filters import Command",
        fixed,
    )
    fixed = re.sub(r",\s*Text\b", "", fixed)
    fixed = re.sub(r"\bText\s*\(\s*\)", "F.text", fixed)

    # register(handler()) -> register(handler, F.text)
    fixed = re.sub(
        r"\.message\.register\s*\(\s*(\w+)\s*\(\s*\)\s*\)",
        r".message.register(\1, F.text)",
        fixed,
    )

    if "F.text" in fixed or "F." in fixed or ".message.register(" in fixed:
        if re.search(r"from\s+aiogram\s+import\s+[^#\n]+", fixed):
            fixed = re.sub(
                r"from\s+aiogram\s+import\s+([^#\n]+)",
                lambda m: (
                    "from aiogram import "
                    + (m.group(1) if "F" in m.group(1) else m.group(1).rstrip() + ", F")
                ).replace(", ,", ",").replace("import ,", "import "),
                fixed,
                count=1,
            )
        elif "from aiogram import" not in fixed:
            fixed = _insert_after_imports(fixed, ["from aiogram import Bot, Dispatcher, F, Router"])

    fixed = re.sub(r"(?<![.\w])exit\s*\(", "sys.exit(", fixed)

    imports_to_add: list[str] = []
    imported = _collect_imported_modules(fixed)
    for pattern, module, import_line in _IMPORT_RULES:
        if re.search(pattern, fixed) and module not in imported:
            imports_to_add.append(import_line)
            imported.add(module.split(".")[0])

    if ("os.getenv" in fixed or "BOT_TOKEN" in fixed) and "load_dotenv" not in fixed:
        imports_to_add.append("from dotenv import load_dotenv")

    if imports_to_add:
        unique_imports: list[str] = []
        for line in imports_to_add:
            if line not in fixed and line not in unique_imports:
                unique_imports.append(line)
        fixed = _insert_after_imports(fixed, unique_imports)

    if "load_dotenv()" not in fixed and "load_dotenv" in fixed:
        fixed = _insert_after_imports(fixed, ["", "load_dotenv()"])

    if "logger = logging.getLogger" not in fixed and "logging." in fixed:
        block = ["", "logging.basicConfig(level=logging.INFO)", "logger = logging.getLogger(__name__)"]
        if "logging.basicConfig" not in fixed:
            fixed = _insert_after_imports(fixed, block)

    if "if not BOT_TOKEN" not in fixed and "BOT_TOKEN = os.getenv" in fixed:
        guard = [
            "",
            "if not BOT_TOKEN:",
            "    sys.exit('BOT_TOKEN не задан. Добавьте его в .env')",
        ]
        # вставляем после присвоения BOT_TOKEN
        lines = fixed.splitlines()
        for idx, line in enumerate(lines):
            if "BOT_TOKEN" in line and "getenv" in line:
                lines = lines[: idx + 1] + guard + lines[idx + 1 :]
                fixed = "\n".join(lines) + "\n"
                break

    fixed = ensure_required_handlers(fixed)
    return fixed.strip() + "\n"


def build_validation_feedback(code: str) -> str:
    issues = validate_bot_code(code)
    if not issues:
        return "Локальная валидация: ошибок не найдено."
    return "Локальная валидация нашла проблемы:\n- " + "\n- ".join(issues)


# ---------------------------------------------------------------------------
# Цепочка:
# analysis_chain → tools_chain → structure_chain → code_chain → review_chain
# ---------------------------------------------------------------------------

def build_chains(llm: OpenAIChat):
    analysis_chain = ANALYSIS_PROMPT | llm.bind(max_tokens=1024) | StrOutputParser()
    tools_chain = TOOLS_PROMPT | llm.bind(max_tokens=1024) | StrOutputParser()
    structure_chain = STRUCTURE_PROMPT | llm.bind(max_tokens=1536) | StrOutputParser()
    code_chain = CODE_PROMPT | llm | StrOutputParser()
    review_chain = REVIEW_PROMPT | llm | StrOutputParser()
    return analysis_chain, tools_chain, structure_chain, code_chain, review_chain


def generate_bot_code(task: str) -> str:
    llm = build_llm()
    (
        analysis_chain,
        tools_chain,
        structure_chain,
        code_chain,
        review_chain,
    ) = build_chains(llm)

    logger.info(
        "Цепочка из 5 шагов; каждый запрос к ProxyAPI может занять до %s с. "
        "Для ускорения используйте PROXY_API_MODEL=gpt-4o-mini",
        int(resolve_proxyapi_timeout()),
    )

    logger.info("Шаг 1/5: анализ задания бота")
    analysis = analysis_chain.invoke({"task": task})
    logger.info("Анализ готов (%s символов)", len(analysis))
    logger.debug("Анализ:\n%s", analysis)

    logger.info("Шаг 2/5: подбор инструментов для генерации")
    tools = tools_chain.invoke({"task": task, "analysis": analysis})
    logger.info("Инструменты подобраны (%s символов)", len(tools))
    logger.debug("Инструменты:\n%s", tools)

    logger.info("Шаг 3/5: создание структуры кода")
    structure = structure_chain.invoke(
        {"task": task, "analysis": analysis, "tools": tools}
    )
    logger.info("Структура готова (%s символов)", len(structure))
    logger.debug("Структура:\n%s", structure)

    logger.info("Шаг 4/5: реализация кода")
    raw_code = code_chain.invoke(
        {
            "task": task,
            "analysis": analysis,
            "tools": tools,
            "structure": structure,
        }
    )
    code = extract_python_code(raw_code)
    code = auto_fix_bot_code(code)
    logger.info("Черновик кода готов (%s символов)", len(code))

    syntax_feedback = build_validation_feedback(code)
    if "ошибок не найдено" not in syntax_feedback:
        logger.warning(syntax_feedback.replace("\n", " | "))
    logger.info("Шаг 5/5: ревью кода")

    reviewed = review_chain.invoke(
        {
            "analysis": analysis,
            "tools": tools,
            "structure": structure,
            "code": code,
            "syntax_feedback": syntax_feedback,
        }
    )
    code = auto_fix_bot_code(extract_python_code(reviewed))

    max_review_passes = 3
    for attempt in range(2, max_review_passes + 1):
        issues = validate_bot_code(code)
        if not issues:
            break
        logger.warning(
            "Повторное ревью (%s/%s): %s",
            attempt,
            max_review_passes,
            "; ".join(issues),
        )
        reviewed = review_chain.invoke(
            {
                "analysis": analysis,
                "tools": tools,
                "structure": structure,
                "code": code,
                "syntax_feedback": build_validation_feedback(code),
            }
        )
        code = auto_fix_bot_code(extract_python_code(reviewed))

    code = auto_fix_bot_code(code)
    final_issues = validate_bot_code(code)
    final_syntax = syntax_error_message(code)

    # Последняя попытка: принудительно добавить /start и /help
    if final_issues and not final_syntax:
        command_issues = {i for i in final_issues if "обработчик" in i or "регистрации" in i}
        if command_issues:
            logger.warning("Принудительная вставка /start и /help")
            code = ensure_required_handlers(code)
            code = auto_fix_bot_code(code)
            final_issues = validate_bot_code(code)
            final_syntax = syntax_error_message(code)

    if final_syntax:
        logger.error("Итоговый код содержит синтаксическую ошибку: %s", final_syntax)
        raise RuntimeError(f"Сгенерированный код содержит синтаксическую ошибку: {final_syntax}")
    if final_issues:
        logger.error("Итоговый код не прошёл валидацию: %s", "; ".join(final_issues))
        raise RuntimeError(
            "Сгенерированный bot.py не прошёл проверку:\n- "
            + "\n- ".join(final_issues)
        )

    logger.info("Генерация завершена, итоговый код: %s символов", len(code))
    return code


def default_output_path() -> Path:
    return Path.cwd() / "bot.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Генерация Telegram-бота по текстовому описанию "
            "(LangChain: анализ → инструменты → структура → код → ревью)."
        )
    )
    parser.add_argument(
        "task",
        help='Текстовое ТЗ, например: "Бот, который отправляет случайные мемы"',
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(default_output_path()),
        help="Путь к итоговому .py файлу (по умолчанию ./bot.py)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Уровень логирования: DEBUG, INFO, WARNING, ERROR (или LOG_LEVEL в .env)",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    task = args.task.strip()
    if not task:
        logger.error("Пустое описание бота")
        return 2

    logger.info("Старт генерации бота по ТЗ: %s", task)
    try:
        code = generate_bot_code(task)
    except Exception:
        logger.exception("Ошибка генерации")
        return 1

    output = Path(args.output)
    output.write_text(code, encoding="utf-8")
    logger.info("Готово: %s", output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
