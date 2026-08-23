import logging
import os
import sys
import time
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv()

PROXYAPI_BASE_URL = "https://openai.api.proxyapi.ru/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
STAGE_COUNT = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class TaskAnalysis(BaseModel):
    category: Literal[
        "техническая поддержка",
        "оплата",
        "заказ",
        "доступ к аккаунту",
        "консультация",
        "жалоба",
        "прочее",
    ] = Field(description="Основная категория обращения")

    priority: Literal["низкий", "средний", "высокий", "критический"] = Field(
        description="Срочность обращения"
    )

    summary: str = Field(
        description="Краткое описание проблемы в одном-двух предложениях"
    )

    entities: list[str] = Field(
        description="Сущности: номер заказа, email, продукт, дата и другие важные данные"
    )

    missing_data: list[str] = Field(
        description="Какие данные нужны дополнительно для решения"
    )

    risks: list[str] = Field(
        description="Риски: персональные данные, финансовые операции, подозрение на мошенничество"
    )


class ToolSelection(BaseModel):
    tools: list[str] = Field(
        description="Инструменты или системы для решения запроса"
    )

    actions: list[str] = Field(
        description="Последовательность действий специалиста"
    )

    escalation_needed: bool = Field(
        description="Нужно ли передать обращение человеку или в другой отдел"
    )

    escalation_reason: str = Field(
        description="Причина эскалации. Если она не нужна, напиши 'Не требуется'."
    )


class ReviewResult(BaseModel):
    approved: bool = Field(
        description="Можно ли отправлять ответ пользователю без доработки"
    )

    issues: list[str] = Field(
        description="Недочёты ответа. Если их нет, верни пустой список."
    )

    final_answer: str = Field(
        description="Исправленная финальная версия ответа для отправки пользователю"
    )


def create_llm() -> ChatOpenAI:
    """Создаёт клиент LLM для ProxyAPI (OpenAI-совместимый API)."""
    api_key = os.getenv("PROXYAPI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Укажите PROXYAPI_KEY в файле .env. "
            "Ключ можно получить на https://console.proxyapi.ru/keys"
        )

    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", DEFAULT_MODEL),
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", PROXYAPI_BASE_URL),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
    )


def log_stage(stage: int, title: str, *, done: bool = False) -> None:
    """Выводит прогресс обработки в консоль и в лог."""
    if done:
        logger.info("[%d/%d] %s — завершено", stage, STAGE_COUNT, title)
        print(f"[{stage}/{STAGE_COUNT}] {title} — готово", flush=True)
        return

    logger.info("[%d/%d] %s — начало", stage, STAGE_COUNT, title)
    print(f"\n[{stage}/{STAGE_COUNT}] {title}...", flush=True)


def read_user_request() -> str:
    """Запрашивает у пользователя текст обращения (поддерживает многострочный ввод)."""
    print("\n=== Обработка обращения в поддержку ===")
    print("Введите текст обращения.")
    print("Для завершения ввода нажмите Enter на пустой строке.\n")

    lines: list[str] = []
    while True:
        try:
            line = input("> " if not lines else "  ")
        except EOFError:
            break

        if not line.strip():
            if lines:
                break
            print("Текст не может быть пустым. Введите обращение:")
            continue

        lines.append(line)

    request = "\n".join(lines).strip()
    if not request:
        raise ValueError("Текст обращения не может быть пустым.")

    logger.info("Получено обращение (%d символов)", len(request))
    print(f"\nПринято обращение ({len(request)} символов). Начинаю обработку...", flush=True)
    return request


def build_chains(llm: ChatOpenAI):
    # Этап 1. Анализ задачи
    analysis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Ты — аналитик службы поддержки.
Проанализируй входящее обращение. Не выдумывай факты.
При определении приоритета учитывай слова о срочности, блокировке доступа,
финансовых сроках и рисках. Верни данные строго по заданной схеме.""",
            ),
            ("human", "Обращение:\n{request}"),
        ]
    )

    analysis_chain = analysis_prompt | llm.with_structured_output(TaskAnalysis)

    # Этап 2. Подбор инструментов
    tools_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Ты — диспетчер процессов поддержки.
На основе анализа выбери только необходимые инструменты и действия.

Доступные инструменты:
- CRM: история клиента, статус обращения, ответственный менеджер;
- База заказов: статус заказа, состав, доставка;
- Платежная система: статус платежа, ссылка на оплату;
- Админ-панель аккаунтов: восстановление доступа, проверка статуса аккаунта;
- База знаний: инструкции и стандартные ответы;
- Эскалация специалисту: ручная проверка, спорная оплата, безопасность.

Не предлагай реально выполнять операции: на этом этапе нужен только план действий.
Верни данные строго по заданной схеме.""",
            ),
            (
                "human",
                """Исходное обращение:
{request}

Результат анализа:
{analysis}""",
            ),
        ]
    )

    tools_chain = tools_prompt | llm.with_structured_output(ToolSelection)

    # Этап 3. Реализация: генерация ответа
    response_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Ты — специалист клиентской поддержки.
Составь короткий, дружелюбный и практичный ответ на русском языке.

Правила:
- Подтверди, что понял суть обращения.
- Дай только те действия, которые можно честно обещать.
- Если не хватает данных — запроси их.
- Не проси пароль, коды из SMS, номер карты целиком, CVV или другие секреты.
- Если нужна эскалация, сообщи, что запрос передан на проверку.
- Не упоминай внутренние названия систем, JSON, ИИ или этапы обработки.""",
            ),
            (
                "human",
                """Обращение клиента:
{request}

Анализ:
{analysis}

План обработки:
{tools}""",
            ),
        ]
    )

    response_chain = response_prompt | llm | StrOutputParser()

    # Этап 4. Проверка и ревью
    review_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Ты — строгий редактор и специалист по качеству поддержки.
Проверь черновик ответа.

Критерии:
1. Ответ соответствует вопросу пользователя.
2. Не содержит выдуманных результатов и необоснованных обещаний.
3. Не запрашивает секретные данные.
4. Вежливый, понятный и содержит следующий шаг.
5. Учитывает срочность и необходимость эскалации.

Если есть проблемы, исправь ответ. Верни результат строго по заданной схеме.""",
            ),
            (
                "human",
                """Обращение:
{request}

Анализ:
{analysis}

Выбранные инструменты:
{tools}

Черновик ответа:
{draft}""",
            ),
        ]
    )

    review_chain = review_prompt | llm.with_structured_output(ReviewResult)

    return analysis_chain, tools_chain, response_chain, review_chain


def process_request(
    request: str,
    analysis_chain,
    tools_chain,
    response_chain,
    review_chain,
) -> ReviewResult:
    """Последовательно выполняет 4 этапа обработки с логированием прогресса."""
    started_at = time.perf_counter()

    log_stage(1, "Анализ обращения")
    stage_started = time.perf_counter()
    analysis = analysis_chain.invoke({"request": request})
    log_stage(1, "Анализ обращения", done=True)
    logger.info(
        "Анализ: категория=%s, приоритет=%s, summary=%s",
        analysis.category,
        analysis.priority,
        analysis.summary,
    )
    print(f"  Категория: {analysis.category}", flush=True)
    print(f"  Приоритет: {analysis.priority}", flush=True)
    print(f"  Суть: {analysis.summary}", flush=True)
    logger.info("Этап 1 выполнен за %.1f с", time.perf_counter() - stage_started)

    analysis_json = analysis.model_dump_json(indent=2)

    log_stage(2, "Подбор инструментов и плана действий")
    stage_started = time.perf_counter()
    tools = tools_chain.invoke({"request": request, "analysis": analysis_json})
    log_stage(2, "Подбор инструментов и плана действий", done=True)
    logger.info(
        "Инструменты: %s; эскалация=%s",
        ", ".join(tools.tools),
        tools.escalation_needed,
    )
    print(f"  Инструменты: {', '.join(tools.tools)}", flush=True)
    print(
        f"  Эскалация: {'да' if tools.escalation_needed else 'нет'}",
        flush=True,
    )
    logger.info("Этап 2 выполнен за %.1f с", time.perf_counter() - stage_started)

    tools_json = tools.model_dump_json(indent=2)

    log_stage(3, "Генерация черновика ответа")
    stage_started = time.perf_counter()
    draft = response_chain.invoke(
        {"request": request, "analysis": analysis_json, "tools": tools_json}
    )
    log_stage(3, "Генерация черновика ответа", done=True)
    logger.info("Черновик сформирован (%d символов)", len(draft))
    print(f"  Черновик: {len(draft)} символов", flush=True)
    logger.info("Этап 3 выполнен за %.1f с", time.perf_counter() - stage_started)

    log_stage(4, "Ревью и финальная проверка")
    stage_started = time.perf_counter()
    result = review_chain.invoke(
        {
            "request": request,
            "analysis": analysis_json,
            "tools": tools_json,
            "draft": draft,
        }
    )
    log_stage(4, "Ревью и финальная проверка", done=True)
    logger.info(
        "Ревью: approved=%s, замечаний=%d",
        result.approved,
        len(result.issues),
    )
    logger.info("Этап 4 выполнен за %.1f с", time.perf_counter() - stage_started)
    logger.info("Обработка завершена за %.1f с", time.perf_counter() - started_at)

    return result


def print_result(result: ReviewResult) -> None:
    print("\n--- РЕЗУЛЬТАТ РЕВЬЮ ---")
    print(f"Можно отправлять: {'да' if result.approved else 'нет'}")

    if result.issues:
        print("\nЗамечания:")
        for issue in result.issues:
            print(f"- {issue}")

    print("\n--- ФИНАЛЬНЫЙ ОТВЕТ ---")
    print(result.final_answer)


def main():
    llm = create_llm()
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    logger.info("Модель: %s", model_name)
    print(f"Подключение к ProxyAPI, модель: {model_name}", flush=True)

    analysis_chain, tools_chain, response_chain, review_chain = build_chains(llm)
    user_request = read_user_request()
    result = process_request(
        user_request,
        analysis_chain,
        tools_chain,
        response_chain,
        review_chain,
    )
    print_result(result)


if __name__ == "__main__":
    main()