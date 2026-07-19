from app.schemas.query import AnswerMode, AnswerResponse, Intent, ParsedQuery, StructuredResult
from app.tools.llm_tool import LLMTool


class AnswerRouter:
    def __init__(self) -> None:
        self.llm = LLMTool()

    def build(self, query: ParsedQuery, result: StructuredResult) -> AnswerResponse:
        if query.answer_mode == AnswerMode.LLM:
            answer_text = self.llm.build_analytical_answer(result)
        else:
            answer_text = self._build_template_answer(query, result)

        return AnswerResponse(
            parsed_query=query,
            result=result,
            answer_mode=query.answer_mode,
            answer_text=answer_text,
            ui_blocks=self._build_ui_blocks(result),
        )

    def _build_template_answer(self, query: ParsedQuery, result: StructuredResult) -> str:
        if result.warnings and not result.rows and not result.metrics and not result.examples:
            return "Не удалось получить данные: " + "; ".join(result.warnings)

        if query.intent == Intent.COUNT_BY_PROBLEM and result.metrics:
            value = result.metrics[0].value
            if query.filters.labels:
                label = query.filters.labels[0]
            elif query.filters.keyword:
                label = f"тексту/товару: {query.filters.keyword}"
            elif query.filters.category:
                label = f"категории: {query.filters.category}"
            else:
                label = "выбранным фильтрам"
            return f"Найдено {value} отзывов по фильтру: {label}."

        if query.intent in {Intent.REVIEW_SAMPLES, Intent.REVIEW_EXAMPLES, Intent.KEYWORD_SEARCH} and result.examples:
            return f"Найдено примеров отзывов: {len(result.examples)}."

        if query.intent == Intent.TOP_PROBLEMS and result.rows:
            top = result.rows[0].data
            examples_part = f" Подобрано примеров: {len(result.examples)}." if result.examples else ""
            return f"Главная проблема: {top.get('label')} — {top.get('review_count')} отзывов.{examples_part}"

        if query.intent == Intent.TOP_PRODUCTS_BY_PROBLEM and result.rows:
            top = result.rows[0].data
            product = top.get("product_name") or top.get("product_id") or "товар без названия"
            problem_reviews = top.get("problem_reviews") or top.get("review_count")
            share = top.get("problem_share_pct")
            if share is not None:
                return f"Больше всего проблемных отзывов у товара: {product} — {problem_reviews} отзывов ({share}% от отзывов товара)."
            return f"Больше всего проблемных отзывов у товара: {product} — {problem_reviews} отзывов."

        if query.intent == Intent.PROBLEM_SHARE and result.rows:
            top = result.rows[0].data
            return f"Самая большая доля: {top.get('label')} — {top.get('share_pct')}%."

        if query.intent in {Intent.PERIOD_COMPARISON, Intent.PROBLEM_GROWTH, Intent.PROBLEM_GROWTH_ANALYSIS} and result.rows:
            top = result.rows[0].data
            return (
                f"Наибольшее изменение: {top.get('label')} — "
                f"{top.get('count_period_1')} → {top.get('count_period_2')} "
                f"(Δ {top.get('delta_abs')}, {top.get('delta_pct')}%)."
            )

        if query.intent == Intent.LABEL_COOCCURRENCE and result.rows:
            top = result.rows[0].data
            return (
                f"Самая частая связка: {top.get('label_1')} + {top.get('label_2')} — "
                f"{top.get('review_count')} отзывов."
            )

        if query.intent == Intent.POSITIVE_VS_PROBLEM and result.rows:
            return f"Найдено периодов: {len(result.rows)}."

        if result.metrics:
            metric = result.metrics[0]
            return f"{metric.name}: {metric.value}"

        if result.rows:
            return f"Найдено строк: {len(result.rows)}"

        if result.examples:
            return f"Найдено примеров отзывов: {len(result.examples)}"

        return "По выбранным фильтрам ничего не найдено."

    def _build_ui_blocks(self, result: StructuredResult) -> list[dict]:
        blocks: list[dict] = []
        if result.metrics:
            blocks.append({"type": "metrics", "items": [m.model_dump() for m in result.metrics]})
        if result.rows:
            blocks.append({"type": "table", "rows": [r.data for r in result.rows]})
        if result.examples:
            blocks.append({"type": "reviews", "items": [e.model_dump() for e in result.examples]})
        if result.warnings:
            blocks.append({"type": "warnings", "items": result.warnings})
        return blocks
