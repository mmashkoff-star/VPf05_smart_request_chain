# Генератор Telegram-бота (LangChain + ProxyAPI)

Python-скрипт, который по текстовому описанию собирает рабочий Telegram-бот на **aiogram 3.x**. Генерация идёт цепочкой LangChain:

анализ задания → подбор инструментов → структура кода → реализация → ревью.

Запросы к моделям OpenAI идут через **[ProxyAPI](https://proxyapi.ru/)**: официальный SDK `openai` с `base_url=https://api.proxyapi.ru/openai/v1`.

## Как устроена цепочка

1. **analysis_chain** — разбирает задание бота: команды, обработчики, формат ответа, БД, зависимости.
2. **tools_chain** — подбирает инструменты для генерации: библиотеки, внешние API, хранение, возможности aiogram 3.x.
3. **structure_chain** — проектирует каркас `bot.py`: импорты, сигнатуры функций, порядок блоков, точка входа.
4. **code_chain** — реализует полный код бота по анализу, инструментам и структуре.
5. **review_chain** — проверяет синтаксис, импорты и точку запуска; при ошибках правит код.

Итог: готовый файл `bot.py`.

Требования к сгенерированному боту:

- aiogram 3.x, обработчики `async def`;
- токен читается из `BOT_TOKEN`;
- код запускается без заглушек и ручных правок.

## Требования

- Python 3.10+
- аккаунт и ключ [ProxyAPI](https://proxyapi.ru/)

## Установка

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Клиент создаётся так:

```python
from openai import OpenAI

client = OpenAI(
    api_key="...",  # PROXY_API_KEY
    base_url="https://api.proxyapi.ru/openai/v1",
)
```

LangChain собирает цепочку из пяти звеньев поверх этого клиента.

## Переменные окружения

Скопируйте пример и заполните ключи:

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

| Переменная | Обязательна | Описание |
|---|---|---|
| `PROXY_API_KEY` | да | ключ ProxyAPI (`OPENAI_API_KEY` тоже принимается) |
| `PROXY_API_BASE_URL` | нет | по умолчанию `https://api.proxyapi.ru/openai/v1` |
| `PROXY_API_MODEL` | нет | модель, по умолчанию `gpt-4o-mini` (`OPENAI_MODEL` — алиас) |
| `OPENAI_MAX_TOKENS` | нет | лимит токенов ответа, по умолчанию `4096` |
| `BOT_TOKEN` | для запуска бота | токен от [@BotFather](https://t.me/BotFather) |
| `LOG_LEVEL` | нет | уровень логов генератора, по умолчанию `INFO` |

## Запуск генерации

```bash
python script.py "Бот, который отправляет случайные мемы"
```

Скрипт создаст `bot.py` в текущей папке. Другой путь:

```bash
python script.py "Бот, который отправляет случайные мемы" -o my_bot.py
python script.py "Бот, который отправляет случайные мемы" --log-level DEBUG
```

Ход цепочки пишется в stderr: шаг 1–5, запросы к ProxyAPI и путь к готовому файлу.

## Запуск сгенерированного бота

```bash
python bot.py
```

Нужен заданный `BOT_TOKEN`. Если в ТЗ были дополнительные пакеты, поставьте их отдельно.
