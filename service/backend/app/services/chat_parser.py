import json
import re
from openai import OpenAI

from app.core.config import get_settings
from app.domain.labels import KNOWN_LABELS, find_labels_in_text
from app.schemas.query import AnswerMode, ChatAskRequest, GroupBy, Intent, ParsedQuery, QuerySource, ReviewFilters, ToolName


_MONTH_RANGES = {
    "сентябр": ("2025-09-01", "2025-09-30"),
    "октябр": ("2025-10-01", "2025-10-31"),
    "ноябр": ("2025-11-01", "2025-11-30"),
    "осен": ("2025-09-01", "2025-11-30"),
}

_SEMANTIC_SCOPE_KEYWORDS = (
    "книг",
    "облож",
    "страниц",
    "учебник",
    "роман",
)


class ChatParser:
    """Chat parser для MVP.

    Если задан OPENAI_API_KEY, сначала пробует LLM → ParsedQuery.
    Если ключа нет или LLM вернул невалидный JSON, использует rule-based fallback.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.last_parse_method = "not_started"
        self.last_parse_error: str | None = None

    def parse(self, request: ChatAskRequest) -> ParsedQuery:
        self.last_parse_error = None
        if self._should_use_rules_fast_path(request.message):
            self.last_parse_method = "rules_fast_path"
            return self._parse_with_rules(request)

        if self.settings.chat_parser_mode != "rules" and self.settings.openai_api_key:
            parsed = self._parse_with_llm(request)
            if parsed is not None:
                self.last_parse_method = "llm"
                return parsed
            self.last_parse_method = "rules_after_llm_failure"
        elif self.settings.chat_parser_mode == "rules":
            self.last_parse_method = "rules_forced"
        else:
            self.last_parse_method = "rules_no_api_key"

        return self._parse_with_rules(request)

    def _parse_with_rules(self, request: ChatAskRequest) -> ParsedQuery:
        text = request.message.lower()

        labels = find_labels_in_text(text)
        date_from, date_to = self._parse_month_range(text)
        keyword = self._parse_keyword(text)
        category = self._parse_category_filter(text)
        group_by = self._parse_group_by(text)
        min_rating, max_rating = self._parse_rating_filters(text)

        intent = self._parse_intent(text, keyword)
        tools = self._tools_for_intent(intent, text)
        answer_mode = AnswerMode.TEMPLATE

        if ToolName.QDRANT in tools:
            answer_mode = AnswerMode.LLM

        if request.force_answer_mode:
            answer_mode = AnswerMode(request.force_answer_mode)

        query = ParsedQuery(
            source=QuerySource.CHAT,
            intent=intent,
            filters=ReviewFilters(
                date_from=date_from,
                date_to=date_to,
                labels=labels,
                keyword=keyword,
                category=category,
                min_rating=min_rating,
                max_rating=max_rating,
            ),
            group_by=group_by,
            semantic_query=request.message,
            tools=tools,
            answer_mode=answer_mode,
        )
        return self._normalize_query(query, request.message)

    def _parse_with_llm(self, request: ChatAskRequest) -> ParsedQuery | None:
        client = OpenAI(api_key=self.settings.openai_api_key, timeout=self.settings.chat_parser_timeout_seconds)
        prompt = self._build_llm_prompt(request.message)

        try:
            response = client.responses.create(
                model=self.settings.openai_model,
                input=prompt,
            )
            data = json.loads(response.output_text)
            data["source"] = QuerySource.CHAT
            data["semantic_query"] = data.get("semantic_query") or request.message

            if request.force_answer_mode:
                data["answer_mode"] = request.force_answer_mode

            query = ParsedQuery.model_validate(data)
            query.filters.labels = [label for label in query.filters.labels if label in KNOWN_LABELS]
            return self._normalize_query(query, request.message)
        except Exception as exc:  # noqa: BLE001
            self.last_parse_error = type(exc).__name__
            return None

    def _build_llm_prompt(self, message: str) -> str:
        labels_json = json.dumps(KNOWN_LABELS, ensure_ascii=False)
        return f"""
Ты parser для сервиса аналитики отзывов маркетплейса.
Преобразуй вопрос пользователя в JSON ParsedQuery. Верни только JSON без markdown.

Доступные labels:
{labels_json}

Допустимые intent:
- count_by_problem
- top_problems
- problem_dynamics
- review_samples
- period_comparison
- problem_share
- problem_growth
- label_cooccurrence
- keyword_search
- positive_vs_problem
- top_products_by_problem
- review_examples

Допустимые tools: postgres, qdrant.
Правила:
1. Для точных чисел, долей, топов, динамики и сравнения периодов используй postgres.
2. Для "похожие отзывы", "примеры", "что именно пишут", "на что жалуются" используй qdrant. Если нужны и числа, и примеры, используй оба tools.
3. Для объяснений, выводов и рекомендаций ставь answer_mode = "llm".
4. Положительный / нейтральный отзыв используй только если пользователь явно спрашивает про позитив/нейтральные отзывы.
5. Не придумывай brand/product/category, если их нет в вопросе.
   Слова вроде "книги", "обложки", "страницы", "учебники" — это semantic scope, а не обязательно точное значение `category`.
   Для таких запросов не ставь `filters.category`, а добавь `qdrant` в tools и положи исходный вопрос в `semantic_query`.
6. Если месяц указан без года, используй 2025 год. сентябрь = 2025-09-01..2025-09-30, октябрь = 2025-10-01..2025-10-31, ноябрь = 2025-11-01..2025-11-30.
7. Если пользователь спрашивает про "рваные", "помятые", "обложки", относись к проблемам качества товара и используй semantic_query как исходный вопрос.
8. Если пользователь спрашивает "главные проблемы у книг", "проблемы книг", "отзывы про книги", используй tools = ["postgres", "qdrant"], intent = "top_problems", answer_mode = "llm".
9. Для "топ товаров", "товары с проблемами", "артикулы с жалобами" используй intent = "top_products_by_problem".
10. Для вопросов "почему", "объясни", "с примерами" вместе с топами/долями/сравнениями используй tools = ["postgres", "qdrant"], answer_mode = "llm".

Форма JSON:
{{
  "source": "chat",
  "intent": "review_examples",
  "filters": {{
    "date_from": null,
    "date_to": null,
    "labels": [],
    "keyword": null,
    "category": null,
    "brand": null,
    "product_id": null,
    "product_name": null,
    "min_rating": null,
    "max_rating": null
  }},
  "group_by": null,
  "semantic_query": "{message}",
  "tools": ["qdrant"],
  "answer_mode": "llm",
  "limit": 20
}}

Вопрос пользователя:
{message}
""".strip()

    def _parse_intent(self, text: str, keyword: str | None) -> Intent:
        if ("товар" in text or "продукт" in text or "артикул" in text) and (
            "топ" in text or "главн" in text or "самые" in text or "больше всего" in text
        ):
            return Intent.TOP_PRODUCTS_BY_PROBLEM
        if self._wants_semantic_search(text):
            return Intent.REVIEW_EXAMPLES
        if "есть ли" in text or "были ли" in text:
            return Intent.COUNT_BY_PROBLEM
        if "было ли" in text or "такое, что" in text:
            return Intent.REVIEW_EXAMPLES
        if keyword or "найди" in text or "поиск" in text:
            return Intent.KEYWORD_SEARCH
        if "покажи отзывы" in text or "тексты отзыв" in text:
            return Intent.REVIEW_SAMPLES
        if "вместе" in text or "связана" in text or "связано" in text or "совмест" in text:
            return Intent.LABEL_COOCCURRENCE
        if "доля" in text or "процент" in text or "структур" in text:
            return Intent.PROBLEM_SHARE
        if "полож" in text and ("проблем" in text or "нейтрал" in text):
            return Intent.POSITIVE_VS_PROBLEM
        if "динамик" in text or "по дня" in text or "по недел" in text or "по месяц" in text:
            return Intent.PROBLEM_DYNAMICS
        if "сравн" in text or "относительно" in text or "что вырос" in text or "что упал" in text or "стало хуже" in text:
            return Intent.PERIOD_COMPARISON
        if "раст" in text or "вырос" in text or "сниз" in text:
            return Intent.PROBLEM_GROWTH
        if "топ" in text or "главн" in text or "самые част" in text:
            return Intent.TOP_PROBLEMS
        return Intent.COUNT_BY_PROBLEM

    def _tools_for_intent(self, intent: Intent, text: str) -> list[ToolName]:
        wants_semantic = self._wants_semantic_search(text)
        wants_examples = any(marker in text for marker in ("пример", "что пишут", "почему", "объясни", "причин"))
        aggregate_intents = {
            Intent.TOP_PROBLEMS,
            Intent.TOP_PRODUCTS_BY_PROBLEM,
            Intent.PROBLEM_SHARE,
            Intent.PERIOD_COMPARISON,
            Intent.PROBLEM_GROWTH,
            Intent.PROBLEM_DYNAMICS,
        }
        if wants_semantic:
            if intent in aggregate_intents:
                return [ToolName.POSTGRES, ToolName.QDRANT]
            return [ToolName.QDRANT]
        if wants_examples and intent in aggregate_intents:
            return [ToolName.POSTGRES]
        if intent == Intent.REVIEW_EXAMPLES:
            if "сколько" in text or "доля" in text or "динамик" in text or "топ" in text:
                return [ToolName.POSTGRES]
            return [ToolName.POSTGRES]
        return [ToolName.POSTGRES]

    def _normalize_query(self, query: ParsedQuery, message: str) -> ParsedQuery:
        query = self._normalize_tools_sql_first(query, message)
        query = self._normalize_semantic_scope(query, message)
        return self._normalize_month_comparison(query, message)

    def _normalize_tools_sql_first(self, query: ParsedQuery, message: str) -> ParsedQuery:
        if self._wants_semantic_search(message.lower()):
            return query
        if ToolName.QDRANT in query.tools:
            query.tools = [tool for tool in query.tools if tool != ToolName.QDRANT]
            if ToolName.POSTGRES not in query.tools:
                query.tools.append(ToolName.POSTGRES)
        query.answer_mode = AnswerMode.TEMPLATE
        return query

    def _normalize_semantic_scope(self, query: ParsedQuery, message: str) -> ParsedQuery:
        text = message.lower()
        category = query.filters.category
        category_is_semantic = bool(category and self._has_semantic_scope(category.lower()))

        if (self._has_semantic_scope(text) or category_is_semantic) and self._wants_semantic_search(text):
            query.filters.category = None if category_is_semantic else category
            query.semantic_query = message
            query.tools = self._merge_tools(query.tools, [ToolName.POSTGRES, ToolName.QDRANT])
            if query.intent in {Intent.TOP_PROBLEMS, Intent.REVIEW_EXAMPLES, Intent.REVIEW_SAMPLES}:
                query.answer_mode = AnswerMode.LLM

        return query

    def _normalize_month_comparison(self, query: ParsedQuery, message: str) -> ParsedQuery:
        if query.intent not in {Intent.PERIOD_COMPARISON, Intent.PROBLEM_GROWTH, Intent.PROBLEM_GROWTH_ANALYSIS}:
            return query

        mentioned_ranges = self._mentioned_month_ranges(message.lower())
        if len(mentioned_ranges) < 2:
            return query

        # SQLBuilder treats date_from/date_to as the second/current period and
        # compares it with the preceding period of equal length. For queries
        # like "сравни октябрь с сентябрем" the current period is October.
        query.filters.date_from, query.filters.date_to = mentioned_ranges[-1]
        return query

    def _has_semantic_scope(self, text: str) -> bool:
        return any(keyword in text for keyword in _SEMANTIC_SCOPE_KEYWORDS)

    def _parse_category_filter(self, text: str) -> str | None:
        if any(marker in text for marker in ("книг", "роман", "учебник")):
            return "Книги"
        return None

    def _wants_semantic_search(self, text: str) -> bool:
        return any(
            marker in text
            for marker in (
                "похож",
                "по смысл",
                "смыслов",
                "что пишут",
                "примеры",
                "пример",
                "найди отзывы похож",
                "похожие отзывы",
            )
        )

    def _should_use_rules_fast_path(self, message: str) -> bool:
        text = message.lower()
        if self._wants_semantic_search(text):
            return True
        return any(
            marker in text
            for marker in (
                "сколько",
                "есть ли",
                "были ли",
                "было ли",
                "такое",
                "что вырос",
                "что упал",
                "динамик",
                "топ",
                "доля",
                "процент",
                "найди",
                "поиск",
                "жалоб",
                "проблем",
            )
        )

    def _merge_tools(self, current: list[ToolName], extra: list[ToolName]) -> list[ToolName]:
        merged = list(current)
        for tool in extra:
            if tool not in merged:
                merged.append(tool)
        return merged

    def _parse_month_range(self, text: str) -> tuple[str | None, str | None]:
        mentioned_ranges = self._mentioned_month_ranges(text)
        if mentioned_ranges:
            return mentioned_ranges[0]
        return None, None

    def _mentioned_month_ranges(self, text: str) -> list[tuple[str, str]]:
        ranges = [date_range for month_stem, date_range in _MONTH_RANGES.items() if month_stem in text]
        return sorted(ranges, key=lambda date_range: date_range[0])

    def _parse_group_by(self, text: str) -> GroupBy | None:
        if "по дня" in text:
            return GroupBy.DAY
        if "по недел" in text:
            return GroupBy.WEEK
        if "по месяц" in text:
            return GroupBy.MONTH
        return None

    def _parse_keyword(self, text: str) -> str | None:
        quoted = re.search(r'["«](.+?)["»]', text)
        if quoted:
            return quoted.group(1).strip()

        match = re.search(r"(?:было ли|были ли)\s+(?:такое,?\s*)?(?:что|когда)?\s*([а-яёa-z0-9 -]{3,80})", text)
        if match:
            return self._normalize_phrase(match.group(1))

        for marker in ("со словом", "слово", "фраза", "где пишут", "где есть"):
            if marker in text:
                tail = text.split(marker, 1)[1].strip(" :—-'")
                if tail:
                    return self._normalize_keyword(tail[:80])
        match = re.search(r"(?:есть ли|были ли)\s+(?:проблемы|жалобы|отзывы)?\s*(?:с|со|про|по|на)?\s+([а-яёa-z0-9 -]{3,60})", text)
        if match:
            return self._normalize_keyword(match.group(1))
        match = re.search(r"(?:проблемы|жалобы|отзывы)\s+(?:с|со|про|по|на)\s+([а-яёa-z0-9 -]{3,60})", text)
        if match:
            return self._normalize_keyword(match.group(1))
        return None

    def _normalize_phrase(self, value: str) -> str:
        raw = value.strip(" ?!.,:;—-\"'«»")
        words = [word for word in re.findall(r"[а-яёa-z0-9]+", raw.lower()) if len(word) > 2]
        return " ".join(words[:8]) if words else raw[:80]

    def _normalize_keyword(self, value: str) -> str:
        raw = value.strip(" ?!.,:;—-\"'«»")
        words = [word for word in re.findall(r"[а-яёa-z0-9]+", raw.lower()) if len(word) > 2]
        if not words:
            return raw[:80]
        word = words[0]
        for suffix in ("ами", "ями", "ами", "ого", "ему", "ыми", "ими", "кой", "кой", "ках", "иях", "ах", "ях", "ой", "ей", "ом", "ем", "а", "я", "ы", "и", "е", "у", "ю"):
            if len(word) > len(suffix) + 4 and word.endswith(suffix):
                return word[: -len(suffix)]
        return word

    def _parse_rating_filters(self, text: str) -> tuple[int | None, int | None]:
        if "низк" in text and "рейтинг" in text:
            return None, 2
        if "негатив" in text or "плохие оценки" in text:
            return None, 2
        match = re.search(r"(?:рейтинг|оценк[аи])\s*(?:<=|до|ниже)\s*([1-5])", text)
        if match:
            return None, int(match.group(1))
        match = re.search(r"(?:рейтинг|оценк[аи])\s*(?:>=|от|выше)\s*([1-5])", text)
        if match:
            return int(match.group(1)), None
        return None, None
