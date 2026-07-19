from openai import OpenAI
from app.core.config import get_settings
from app.schemas.query import StructuredResult


class LLMTool:
    """Финальный аналитический слой.

    Получает только структурированные данные, а не полный датасет.
    Это снижает риск галлюцинаций: LLM объясняет уже посчитанные факты.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def build_analytical_answer(self, result: StructuredResult) -> str:
        if result.warnings and not result.rows and not result.metrics and not result.examples:
            return self._no_data_answer(result)

        if not self.settings.openai_api_key:
            return self._fallback_answer(result)

        client = OpenAI(api_key=self.settings.openai_api_key, timeout=self.settings.openai_timeout_seconds)
        prompt = self._build_prompt(result)

        try:
            response = client.responses.create(
                model=self.settings.openai_model,
                input=prompt,
            )
            return response.output_text
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"LLM summary не успел или недоступен: {type(exc).__name__}.")
            return self._fallback_answer(result)

    def _build_prompt(self, result: StructuredResult) -> str:
        return f"""
Ты аналитик сервиса отзывов маркетплейса.
Сделай краткий вывод по структурированным данным.

Правила:
- не придумывай причин, которых нет в данных;
- не придумывай числа, товары, бренды или категории; используй только Metrics, Rows и Examples ниже;
- LLM не считает по отзывам сама, а только интерпретирует уже посчитанные PostgreSQL числа и найденные примеры;
- Examples используй только как иллюстрации паттернов; не выводи по ним общую статистику и доли;
- отделяй факт от гипотезы;
- пиши коротко и прикладно для продавца;
- если данных мало, прямо скажи об этом.
- не используй слово "разметка" в ответе пользователю; говори "данные", "отзывы", "labels" или "классы проблем";
- не используй внутренние термины "semantic-кандидаты", "RAG-кандидаты", "PostgreSQL-топ", "пайплайн" в пользовательском выводе;
- если запрос использовал смысловой поиск, говори проще: "искал отзывы по смыслу";
- если есть warnings, переведи их на язык пользователя и не показывай технические детали в основном выводе.
- если есть и Rows, и Examples, сначала дай проверяемые цифры, затем 2-4 характерных примера.
- добавь коротко: что это значит для продавца, что проверить дополнительно, ограничения данных.

ParsedQuery:
{result.parsed_query.model_dump_json(indent=2)}

Metrics:
{[m.model_dump() for m in result.metrics]}

Rows:
{[r.data for r in result.rows[:30]]}

Examples:
{[e.model_dump() for e in result.examples[:10]]}

Warnings:
{result.warnings}

Data coverage / raw notes:
{result.raw.get("data_coverage")}
""".strip()

    def _no_data_answer(self, result: StructuredResult) -> str:
        uses_semantic_search = bool(result.parsed_query.semantic_query)
        if uses_semantic_search:
            return (
                "Не нашел достаточно релевантных отзывов для такого вопроса. "
                "Попробуйте уточнить формулировку, выбрать более широкий период или задать товар/категорию точнее."
            )
        return (
            "По выбранным фильтрам данных не найдено. "
            "Попробуйте расширить период или убрать часть фильтров."
        )

    def _fallback_answer(self, result: StructuredResult) -> str:
        if result.warnings:
            return self._no_data_answer(result)
        if result.metrics:
            parts = [f"{m.name}: {m.value}" for m in result.metrics]
            return "Краткий результат: " + ", ".join(parts) + ". Для аналитического вывода подключи OPENAI_API_KEY."
        if result.rows:
            return f"Найдено строк: {len(result.rows)}. Для аналитического вывода подключи OPENAI_API_KEY."
        if result.examples:
            return f"Найдено примеров отзывов: {len(result.examples)}. Для аналитического вывода подключи OPENAI_API_KEY."
        return "Данных для вывода пока нет. Проверь фильтры и подключение инструментов."
