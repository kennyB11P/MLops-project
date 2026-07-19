import re
import json
from time import perf_counter

from app.schemas.query import (
    AnswerMode,
    AnswerResponse,
    ChatAskRequest,
    Intent,
    ParsedQuery,
    QuerySource,
    ReviewExample,
    StructuredResult,
    TemplateExecuteRequest,
    ToolName,
    TraceStep,
)
from app.core.config import get_settings
from app.domain.labels import POSITIVE_LABEL, PROBLEM_LABELS
from app.services.answer_router import AnswerRouter
from app.services.chat_parser import ChatParser
from app.services.template_parser import TemplateParser
from app.services.ttl_cache import TTLCache
from app.services.tool_router import ToolRouter


class QueryService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.template_parser = TemplateParser()
        self.chat_parser = ChatParser()
        self.tool_router = ToolRouter()
        self.answer_router = AnswerRouter()
        self.response_cache = TTLCache[AnswerResponse](ttl_seconds=self.settings.cache_ttl_seconds, max_size=128)

    def execute_parsed_query(self, query: ParsedQuery, trace_steps: list[TraceStep] | None = None) -> AnswerResponse:
        trace_steps = trace_steps or []
        started_at = perf_counter()

        tools_started_at = perf_counter()
        structured_result = self.tool_router.run(query)
        tool_steps = structured_result.raw.pop("trace_steps", [])
        trace_steps.append(
            TraceStep(
                id="tools",
                title="Выполнение инструментов",
                duration_ms=int((perf_counter() - tools_started_at) * 1000),
                input={"tools": query.tools, "intent": query.intent},
                output={
                    "metrics_count": len(structured_result.metrics),
                    "rows_count": len(structured_result.rows),
                    "examples_count": len(structured_result.examples),
                    "warnings_count": len(structured_result.warnings),
                },
            )
        )
        trace_steps.extend(TraceStep.model_validate(step) for step in tool_steps)

        examples_started_at = perf_counter()
        examples_step = self._attach_top_problem_examples(query, structured_result)
        if examples_step:
            examples_step.duration_ms = int((perf_counter() - examples_started_at) * 1000)
            trace_steps.append(examples_step)
        count_examples_step = self._attach_chat_filter_examples(query, structured_result)
        if count_examples_step:
            count_examples_step.duration_ms = int((perf_counter() - examples_started_at) * 1000)
            trace_steps.append(count_examples_step)

        answer_started_at = perf_counter()
        response = self.answer_router.build(query, structured_result)
        trace_steps.append(
            TraceStep(
                id="answer",
                title="Формирование ответа",
                duration_ms=int((perf_counter() - answer_started_at) * 1000),
                input={"answer_mode": query.answer_mode},
                output={"answer_text": response.answer_text, "ui_blocks_count": len(response.ui_blocks)},
            )
        )
        response.execution_ms = int((perf_counter() - started_at) * 1000)
        response.trace_steps = trace_steps
        response.result.raw["latency_steps"] = [
            {"id": step.id, "title": step.title, "duration_ms": step.duration_ms, "status": step.status}
            for step in trace_steps
        ]
        return response

    def execute_template(self, template_id: str, request: TemplateExecuteRequest) -> AnswerResponse:
        cache_key = self._cache_key("template", {"template_id": template_id, "request": request.model_dump()})
        cached = self.response_cache.get(cache_key)
        if cached:
            cached.result.raw["cache_hit"] = True
            cached.trace_steps.insert(
                0,
                TraceStep(
                    id="response_cache",
                    title="Ответ из in-memory cache",
                    duration_ms=0,
                    input={"cache_key": cache_key},
                    output={"hit": True},
                ),
            )
            return cached

        started_at = perf_counter()
        parsed_query = self.template_parser.parse(template_id, request)
        trace_steps = [
            TraceStep(
                id="parse",
                title="Разбор формы в ParsedQuery",
                duration_ms=int((perf_counter() - started_at) * 1000),
                input={"template_id": template_id, "request": request.model_dump()},
                output={"parsed_query": parsed_query.model_dump()},
            )
        ]
        response = self.execute_parsed_query(parsed_query, trace_steps)
        self.response_cache.set(cache_key, response)
        return response

    def _attach_top_problem_examples(self, query: ParsedQuery, result: StructuredResult) -> TraceStep | None:
        if query.intent != Intent.TOP_PROBLEMS or query.examples_limit <= 0:
            return None

        labels = self._labels_for_top_problem_examples(query, result)
        if not labels:
            result.raw["top_problem_examples"] = {
                "labels": [],
                "candidate_count": 0,
                "selected_count": 0,
                "diversity_method": "none",
            }
            return TraceStep(
                id="top_problem_examples",
                title="Подбор примеров для топа проблем",
                input={"examples_limit": query.examples_limit},
                output={"selected_count": 0, "reason": "label_not_found"},
            )

        candidate_limit = min(100, max(30, query.examples_limit * 5))
        sample_query = query.model_copy(deep=True)
        sample_query.intent = Intent.REVIEW_SAMPLES
        sample_query.tools = [ToolName.POSTGRES]
        sample_query.answer_mode = AnswerMode.TEMPLATE
        sample_query.limit = candidate_limit
        sample_query.examples_limit = 0
        sample_query.filters.labels = labels

        sample_result = self.tool_router.postgres.run(sample_query)
        for warning in sample_result.warnings:
            if warning not in result.warnings and ("Ошибка" in warning or "POSTGRES_DSN" in warning):
                result.warnings.append(warning)

        candidates = [example for example in sample_result.examples if example.text.strip()]
        selected = self._select_diverse_examples(candidates, labels, query.examples_limit)
        result.examples = selected
        result.raw["top_problem_examples"] = {
            "labels": labels,
            "candidate_limit": candidate_limit,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "diversity_method": "greedy_token_jaccard" if len(candidates) > 5 else "informative_order",
        }

        return TraceStep(
            id="top_problem_examples",
            title="Подбор примеров для топа проблем",
            input={"labels": labels, "examples_limit": query.examples_limit, "candidate_limit": candidate_limit},
            output=result.raw["top_problem_examples"],
        )

    def _labels_for_top_problem_examples(self, query: ParsedQuery, result: StructuredResult) -> list[str]:
        if query.filters.labels:
            return list(query.filters.labels)

        for row in result.rows:
            label = row.data.get("label")
            if isinstance(label, str) and label.strip():
                return [label]
        return []

    def _attach_chat_filter_examples(self, query: ParsedQuery, result: StructuredResult) -> TraceStep | None:
        if query.source != QuerySource.CHAT:
            return None
        if query.intent not in {Intent.COUNT_BY_PROBLEM, Intent.KEYWORD_SEARCH}:
            return None
        if result.examples or not self._has_example_worthy_filters(query):
            return None

        sample_query = query.model_copy(deep=True)
        sample_query.intent = Intent.REVIEW_SAMPLES
        sample_query.tools = [ToolName.POSTGRES]
        sample_query.answer_mode = AnswerMode.TEMPLATE
        sample_query.limit = min(max(query.examples_limit or 5, 5), 10)
        sample_query.examples_limit = 0
        if query.intent == Intent.COUNT_BY_PROBLEM and not sample_query.filters.labels:
            sample_query.filters.labels = PROBLEM_LABELS

        sample_result = self.tool_router.postgres.run(sample_query)
        candidates = [example for example in sample_result.examples if example.text.strip()]
        selected_limit = min(len(candidates), 5)
        result.examples = self._select_diverse_examples(candidates, query.filters.labels, selected_limit)
        result.raw["chat_filter_examples"] = {
            "candidate_count": len(candidates),
            "selected_count": len(result.examples),
            "source": "postgres_review_samples",
        }
        return TraceStep(
            id="chat_filter_examples",
            title="Подбор примеров по фильтрам чата",
            input={"intent": query.intent, "filters": query.filters.model_dump(), "limit": sample_query.limit},
            output=result.raw["chat_filter_examples"],
        )

    def _has_example_worthy_filters(self, query: ParsedQuery) -> bool:
        return bool(
            query.filters.labels
            or query.filters.keyword
            or query.filters.category
            or query.filters.brand
            or query.filters.product_id
            or query.filters.product_name
        )

    def _select_diverse_examples(
        self,
        candidates: list[ReviewExample],
        labels: list[str],
        examples_limit: int,
    ) -> list[ReviewExample]:
        if examples_limit <= 0:
            return []

        scored = sorted(candidates, key=lambda example: self._example_base_score(example, labels), reverse=True)
        if len(scored) <= examples_limit:
            return scored
        if len(scored) <= 5:
            return scored[:examples_limit]

        selected: list[ReviewExample] = []
        remaining = scored[:]
        while remaining and len(selected) < examples_limit:
            best = max(
                remaining,
                key=lambda example: self._example_diversity_score(example, labels, selected),
            )
            selected.append(best)
            remaining.remove(best)
        return selected

    def _example_base_score(self, example: ReviewExample, labels: list[str]) -> float:
        text_length = len(example.text.strip())
        score = 0.0
        if text_length >= 30:
            score += 2.0
        if text_length >= 80:
            score += 1.0
        if text_length <= 700:
            score += 0.4
        if set(labels).intersection(example.labels):
            score += 2.0
        if labels and POSITIVE_LABEL in example.labels:
            score -= 1.5
        if example.product_id:
            score += 0.3
        if example.product_name:
            score += 0.3
        if example.rating is not None:
            score += 0.2
        if example.date:
            score += 0.2
        return score

    def _example_diversity_score(
        self,
        example: ReviewExample,
        labels: list[str],
        selected: list[ReviewExample],
    ) -> float:
        score = self._example_base_score(example, labels)
        if not selected:
            return score

        max_similarity = max(self._text_similarity(example.text, item.text) for item in selected)
        selected_product_ids = {item.product_id for item in selected if item.product_id}
        selected_ratings = {item.rating for item in selected if item.rating is not None}
        selected_dates = {item.date for item in selected if item.date}

        if example.product_id and example.product_id not in selected_product_ids:
            score += 0.7
        if example.rating is not None and example.rating not in selected_ratings:
            score += 0.35
        if example.date and example.date not in selected_dates:
            score += 0.25

        return score - max_similarity * 3.0

    def _text_similarity(self, left: str, right: str) -> float:
        left_tokens = self._text_tokens(left)
        right_tokens = self._text_tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _text_tokens(self, text: str) -> set[str]:
        return set(re.findall(r"[0-9a-zа-яё]{3,}", text.lower()))

    def ask_chat(self, request: ChatAskRequest) -> AnswerResponse:
        cache_key = self._cache_key("chat", request.model_dump())
        cached = self.response_cache.get(cache_key)
        if cached:
            cached.result.raw["cache_hit"] = True
            cached.trace_steps.insert(
                0,
                TraceStep(
                    id="response_cache",
                    title="Ответ из in-memory cache",
                    duration_ms=0,
                    input={"cache_key": cache_key},
                    output={"hit": True},
                ),
            )
            return cached

        started_at = perf_counter()
        parsed_query = self.chat_parser.parse(request)
        trace_steps = [
            TraceStep(
                id="parse",
                title="Разбор вопроса в ParsedQuery",
                duration_ms=int((perf_counter() - started_at) * 1000),
                input={"message": request.message, "force_answer_mode": request.force_answer_mode},
                output={
                    "parse_method": self.chat_parser.last_parse_method,
                    "parse_error": self.chat_parser.last_parse_error,
                    "llm_model": self.chat_parser.settings.openai_model
                    if self.chat_parser.last_parse_method == "llm"
                    else None,
                    "parsed_query": parsed_query.model_dump(),
                },
            )
        ]
        response = self.execute_parsed_query(parsed_query, trace_steps)
        self.response_cache.set(cache_key, response)
        return response

    def _cache_key(self, namespace: str, payload: dict) -> str:
        return namespace + ":" + json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
