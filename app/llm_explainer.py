import json
import re

from openai import OpenAI

from app.config import (
    LLM_MAX_OUTPUT_TOKENS,
    POLZA_API_KEY,
    POLZA_BASE_URL,
    POLZA_MODEL,
)


def is_llm_configured() -> bool:
    api_key = POLZA_API_KEY.strip()
    return bool(api_key and api_key != "your_polza_api_key_here")


def build_explanation_prompt(values: dict, prediction: dict) -> str:
    probabilities = prediction.get("probabilities", {})
    model_probability = probabilities.get(prediction["prediction"])
    public_payload = {
        "input_values": values,
        "prediction": prediction["prediction"],
        "prediction_probability": model_probability,
        "probabilities": probabilities,
        "best_model": prediction["model_name"],
        "most_important_features": prediction.get("top_features", []),
    }
    return json.dumps(public_payload, ensure_ascii=False, indent=2)


def explain_prediction(values: dict, prediction: dict) -> str:
    if not is_llm_configured():
        return (
            "LLM-пояснение не сформировано: задайте POLZA_API_KEY "
            "в переменных окружения или в файле .env."
        )

    client = OpenAI(
        api_key=POLZA_API_KEY,
        base_url=POLZA_BASE_URL,
    )

    instructions = (
        "Ты объясняешь результат ML-классификатора девиантного риска для учебного "
        "FastAPI-приложения. Используй только переданные данные: входные признаки, "
        "вероятности классов, название модели прогнозирования и список "
        "наиболее важных признаков. Не выдавай медицинских, юридических или "
        "психологических диагнозов. Не характеризуй приложение как экспериментальную, "
        "черновую или предварительную версию. Не упоминай accuracy, Macro F1, "
        "Weighted F1, точность модели и любые проценты качества обучения. "
        "Пиши по-русски, простым человеческим языком, без Markdown-разметки, без "
        "звездочек, без маркированных списков и без нумерации. Ответ должен состоять "
        "из блока сравнения и одного короткого абзаца описания. Блок сравнения "
        "оформи строго в 5 отдельных строк: "
        "\"Вердикт LLM: <класс>\"; "
        "\"Уверенность LLM: <процент>%\"; "
        "\"Прогноз <best_model>: <prediction>\"; "
        "\"Вероятность модели: <prediction_probability в процентах>%\"; "
        "\"Сравнение: <совпадают ли выводы LLM и модели>\". "
        "Вероятность модели переводи из доли в проценты: 0.8779 нужно писать как "
        "87,79%. "
        "Уверенность LLM оцени самостоятельно по согласованности входных признаков "
        "и близости вероятностей классов; это отдельная оценка LLM, а не метрика "
        "обучения модели. После блока сравнения через пустую строку дай описание "
        "причин прогноза. Не "
        "перечисляй признаки как сухой список значений: связывай их в логичное "
        "пояснение. Например, объясняй, что низкая частота "
        "тревожных текстовых признаков, отсутствие деструктивных хэштегов и умеренные "
        "показатели активности вместе поддерживают выбранный класс. Упоминай числовые "
        "значения только там, где они помогают понять вывод. Не добавляй финальную "
        "оговорку о предварительном, учебном или не окончательном заключении."
    )

    response = client.chat.completions.create(
        model=POLZA_MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": build_explanation_prompt(values, prediction)},
        ],
        max_tokens=LLM_MAX_OUTPUT_TOKENS,
    )
    content = response.choices[0].message.content or ""
    return clean_explanation_text(content)


def clean_explanation_text(text: str) -> str:
    text = re.sub(r"(?m)^\s*[*\-•]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+[\).]\s+", "", text)
    text = re.sub(r"(?im)^.*(?:учебн|статистическ|окончательн(?:ое|ым)? заключен).*$(?:\r?\n)?", "", text)
    text = format_comparison_block(text)
    text = text.replace("**", "")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_comparison_block(text: str) -> str:
    pattern = re.compile(
        r"^\s*Вердикт LLM:\s*(?P<llm_class>[^,\n.]+),\s*"
        r"уверенность LLM:\s*(?P<llm_confidence>[^.\n]+)\.\s*"
        r"Прогноз\s+(?P<model_name>[^:]+):\s*(?P<model_class>[^,\n.]+),\s*"
        r"вероятность модели:\s*(?P<model_probability>[^.\n]+)\.\s*"
        r"(?P<comparison>Выводы[^\n.]*\.)",
        flags=re.IGNORECASE,
    )

    def replacement(match: re.Match) -> str:
        return (
            f"Вердикт LLM: {match.group('llm_class').strip()}\n"
            f"Уверенность LLM: {match.group('llm_confidence').strip()}\n"
            f"Прогноз {match.group('model_name').strip()}: {match.group('model_class').strip()}\n"
            f"Вероятность модели: {match.group('model_probability').strip()}\n"
            f"Сравнение: {match.group('comparison').strip()}"
        )

    return pattern.sub(replacement, text, count=1)
