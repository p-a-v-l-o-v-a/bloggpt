"""
app.py — FastAPI-сервис генерации блог-постов на основе актуальных новостей.

Функциональность:
- Получение свежих новостей по теме через Currents API
- Генерация заголовка, meta description и статьи через OpenAI ChatCompletion API
- Обработка ошибок и проверка переменных окружения
- Health endpoints для мониторинга
- Запуск через uvicorn
"""

import os
import logging
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import openai  # старый стиль библиотеки

# =========================
#   ЛОГГИРОВАНИЕ
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# =========================
#   ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "Переменная окружения OPENAI_API_KEY не установлена. "
        "Задайте её перед запуском сервиса."
    )

if not CURRENTS_API_KEY:
    raise RuntimeError(
        "Переменная окружения CURRENTS_API_KEY не установлена. "
        "Задайте её перед запуском сервиса."
    )

# Инициализируем ключ для старого клиента openai
openai.api_key = OPENAI_API_KEY

# =========================
#   ИНИЦИАЛИЗАЦИЯ FASTAPI
# =========================

app = FastAPI(
    title="Blog Post Generator",
    description="Сервис генерации блог-постов на основе актуальных новостей (Currents API + OpenAI).",
    version="1.0.0",
)

# =========================
#   Pydantic-МОДЕЛИ
# =========================


class TopicRequest(BaseModel):
    """Модель входных данных с темой поста."""

    topic: str = Field(..., description="Тема для генерации статьи.")


class GeneratedPostResponse(BaseModel):
    """Модель выходных данных с результатом генерации."""

    title: str
    meta_description: str
    post_content: str
    raw_news_context: Optional[str] = Field(
        None,
        description="Сырые новости, которые использовались как контекст (для отладки).",
    )


class HealthResponse(BaseModel):
    """Ответ для health-эндпоинтов."""

    status: str
    message: Optional[str] = None


# =========================
#   ПОЛУЧЕНИЕ НОВОСТЕЙ ИЗ CURRENTS API
# =========================


def get_recent_news(topic: str, language: str = "en", limit: int = 5) -> str:
    """
    Получает свежие новости по заданной теме из Currents API и
    возвращает их в виде текстового блока, пригодного для передачи в модель.

    :param topic: Тема/ключевые слова для поиска новостей.
    :param language: Язык новостей (по умолчанию английский).
    :param limit: Максимальное количество новостных статей в контексте.
    :return: Отформатированная строка с новостями.
    """
    url = "https://api.currentsapi.services/v1/latest-news"
    params = {
        "language": language,
        "keywords": topic,
        "apiKey": CURRENTS_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=15)
    except requests.RequestException as e:
        logger.error(f"Ошибка сети при обращении к Currents API: {e}")
        raise HTTPException(
            status_code=502,
            detail="Ошибка при обращении к Currents API (проблема сети или таймаут).",
        )

    if response.status_code != 200:
        logger.error(
            "Currents API вернул ошибку. "
            f"Статус: {response.status_code}, ответ: {response.text}"
        )
        raise HTTPException(
            status_code=502,
            detail=f"Currents API вернул ошибку: {response.text}",
        )

    data = response.json()
    news_data = data.get("news", [])

    if not news_data:
        logger.info(f"Новости по теме '{topic}' не найдены.")
        return "Свежих новостей по этой теме не найдено."

    # Формируем текстовый контекст из первых `limit` статей
    formatted_news_parts = []
    for article in news_data[:limit]:
        title = article.get("title", "Без заголовка")
        description = article.get("description") or ""
        source = article.get("source") or "неизвестный источник"
        published = article.get("published") or "дата не указана"

        formatted_news_parts.append(
            f"- {title} ({source}, {published})\n  {description}"
        )

    formatted_news = "\n".join(formatted_news_parts)
    logger.info(f"Получено {min(len(news_data), limit)} новостей для темы '{topic}'.")
    return formatted_news


# =========================
#   ГЕНЕРАЦИЯ КОНТЕНТА ЧЕРЕЗ OPENAI
# =========================


def generate_content(topic: str) -> GeneratedPostResponse:
    """
    Генерирует заголовок, meta-описание и тело статьи по теме,
    используя актуальные новости как контекст для OpenAI.

    :param topic: Тема статьи.
    :return: Объект GeneratedPostResponse с результатами генерации.
    """
    # 1. Получаем свежие новости по теме
    recent_news_context = get_recent_news(topic)

    try:
        # 2. Генерация заголовка статьи
        logger.info(f"Запрос к OpenAI для генерации заголовка по теме '{topic}'.")
        title_completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # можно заменить на gpt-4.1-mini, если доступен
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Придумайте привлекательный и точный заголовок для статьи "
                        f"на тему '{topic}', с учётом актуальных новостей:\n"
                        f"{recent_news_context}\n\n"
                        "Заголовок должен быть интересным, живым и ясно передавать суть темы. "
                        "Ответьте только заголовком, без кавычек и лишнего текста."
                    ),
                }
            ],
            max_tokens=60,
            temperature=0.5,
        )
        title = title_completion.choices[0].message["content"].strip()

        # 3. Генерация meta-описания
        logger.info("Запрос к OpenAI для генерации meta-описания.")
        meta_completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Напишите meta-описание для статьи с заголовком: "
                        f"'{title}'. Описание должно быть полным, информативным, "
                        "включать основные ключевые слова по теме и быть длиной 140–160 символов. "
                        "Не добавляйте ничего лишнего, просто одно законченное предложение."
                    ),
                }
            ],
            max_tokens=120,
            temperature=0.5,
        )
        meta_description = meta_completion.choices[0].message["content"].strip()

        # 4. Генерация полного текста статьи
        logger.info("Запрос к OpenAI для генерации полного текста статьи.")
        post_completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Напишите подробную статью на тему '{topic}', используя следующие "
                        f"актуальные новости как контекст:\n\n"
                        f"{recent_news_context}\n\n"
                        "Требования к статье:\n"
                        "1. Статья должна быть информативной, логичной и структурированной.\n"
                        "2. Объём — не менее 1500 символов.\n"
                        "3. Обязательно используйте подзаголовки (h2/h3 в формате текста, например: '## Подзаголовок').\n"
                        "4. Включите анализ текущих трендов, отражённых в новостях.\n"
                        "5. Сделайте явные блоки: вступление, основная часть и заключение.\n"
                        "6. Приведите примеры из актуальных новостей, упомянутых выше (без прямых цитат).\n"
                        "7. Каждый абзац — минимум 3–4 предложения.\n"
                        "8. Текст должен быть лёгким для восприятия и полезным для читателя.\n"
                        "9. Пишите по-русски, но кратко поясняйте англоязычные термины, если используете их.\n"
                    ),
                }
            ],
            max_tokens=1500,
            temperature=0.6,
            presence_penalty=0.6,
            frequency_penalty=0.6,
        )
        post_content = post_completion.choices[0].message["content"].strip()

        return GeneratedPostResponse(
            title=title,
            meta_description=meta_description,
            post_content=post_content,
            raw_news_context=recent_news_context,
        )

    except HTTPException:
        # Если внутри уже был брошен HTTPException (например, в get_recent_news) — просто прокидываем дальше
        raise
    except Exception as e:
        logger.exception(f"Неожиданная ошибка при генерации контента: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при генерации контента: {str(e)}",
        )


# =========================
#   ЭНДПОИНТЫ FASTAPI
# =========================


@app.post(
    "/generate-post",
    response_model=GeneratedPostResponse,
    summary="Сгенерировать блог-пост по теме",
    description="Генерирует заголовок, meta description и полный текст статьи на основе темы и свежих новостей.",
)
async def generate_post_api(payload: TopicRequest):
    """
    Основной эндпоинт: принимает тему и возвращает сгенерированный пост.
    """
    logger.info(f"Запрос на генерацию поста по теме: '{payload.topic}'")
    return generate_content(payload.topic)


@app.get(
    "/",
    response_model=HealthResponse,
    summary="Базовый статус сервиса",
    description="Простой эндпоинт для проверки, что сервис запущен.",
)
async def root():
    """
    Корневой эндпоинт для быстрой проверки работоспособности сервиса.
    """
    return HealthResponse(status="OK", message="Service is running")


@app.get(
    "/heartbeat",
    response_model=HealthResponse,
    summary="Heartbeat",
    description="Эндпоинт для проверки 'живости' сервиса (используется мониторингом).",
)
async def heartbeat_api():
    """
    Эндпоинт для проверки состояния сервиса (heartbeat).
    """
    return HealthResponse(status="OK", message="Heartbeat is fine")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Расширенный health-check",
    description="Проверка того, что доступны ключевые переменные окружения и базовая логика.",
)
async def health_check():
    """
    Эндпоинт расширенной проверки: валидирует наличие ключей и возвращает статус.
    """
    errors = []
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is missing")
    if not CURRENTS_API_KEY:
        errors.append("CURRENTS_API_KEY is missing")

    if errors:
        return HealthResponse(status="ERROR", message="; ".join(errors))

    return HealthResponse(status="OK", message="All required configs are present")


# =========================
#   ЗАПУСК ЧЕРЕЗ UVICORN
# =========================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))

    # Важно: файл называется app.py — указываем "app:app"
    uvicorn.run("app:app", host="0.0.0.0", port=port)

