from collections import Counter
from time import perf_counter

from app.domain.labels import POSITIVE_LABEL
from app.schemas.query import AnswerMode, Intent, ParsedQuery, ResultRow, StructuredResult, ToolName, TraceStep
from app.tools.postgres_tool import PostgresTool
from app.tools.qdrant_tool import QdrantTool


class ToolRouter:
    """Вызывает нужные инструменты по списку query.tools."""

    def __init__(self) -> None:
        self.postgres = PostgresTool()
        self.qdrant = QdrantTool()

    def run(self, query: ParsedQuery) -> StructuredResult:
        merged = StructuredResult(parsed_query=query)

        if ToolName.POSTGRES in query.tools:
            started_at = perf_counter()
            postgres_result = self.postgres.run(query)
            self._merge(merged, postgres_result)
            self._add_tool_step(merged, ToolName.POSTGRES, query, postgres_result, started_at)

        if ToolName.QDRANT in query.tools:
            started_at = perf_counter()
            qdrant_result = self.qdrant.run(query)
            self._merge(merged, qdrant_result)
            self._add_tool_step(merged, ToolName.QDRANT, query, qdrant_result, started_at)
            if qdrant_result.warnings and not qdrant_result.examples and ToolName.POSTGRES not in query.tools:
                self._run_postgres_fallback(query, merged)

        if not query.tools:
            merged.warnings.append("В ParsedQuery не указан ни один инструмент.")

        self._apply_rag_aggregates(query, merged)

        return merged

    def _run_postgres_fallback(self, query: ParsedQuery, merged: StructuredResult) -> None:
        has_fallback_filter = bool(
            query.filters.labels
            or query.filters.keyword
            or query.filters.product_id
            or query.filters.product_name
            or query.filters.category
            or query.filters.brand
        )
        if not has_fallback_filter:
            merged.warnings.append(
                "Смысловой поиск не успел, а точного label/keyword-фильтра для PostgreSQL fallback нет."
            )
            return

        fallback_query = query.model_copy(deep=True)
        fallback_query.intent = Intent.REVIEW_SAMPLES
        fallback_query.tools = [ToolName.POSTGRES]
        fallback_query.answer_mode = AnswerMode.TEMPLATE
        started_at = perf_counter()
        fallback_result = self.postgres.run(fallback_query)
        fallback_result.warnings.append(
            "Смысловой поиск не успел; показан PostgreSQL fallback по распознанным labels/keyword/фильтрам."
        )
        self._merge(merged, fallback_result)
        self._add_tool_step(merged, ToolName.POSTGRES, fallback_query, fallback_result, started_at)

    def _merge(self, target: StructuredResult, source: StructuredResult) -> None:
        target.metrics.extend(source.metrics)
        target.rows.extend(source.rows)
        target.examples.extend(source.examples)
        target.warnings.extend(source.warnings)
        trace_steps = target.raw.get("trace_steps", [])
        target.raw.update(source.raw)
        if trace_steps:
            source_trace_steps = source.raw.get("trace_steps", [])
            target.raw["trace_steps"] = [*trace_steps, *source_trace_steps]

    def _apply_rag_aggregates(self, query: ParsedQuery, result: StructuredResult) -> None:
        if ToolName.QDRANT not in query.tools:
            return

        if query.intent == Intent.TOP_PROBLEMS and query.semantic_query and not result.examples:
            result.rows = []
            result.metrics = []
            result.warnings.append(
                "Не удалось найти релевантные отзывы по смысловому запросу. Общая статистика без таких отзывов не показана, чтобы не исказить вывод."
            )
            return

        if not result.examples:
            return

        if query.intent == Intent.TOP_PROBLEMS:
            counts: Counter[str] = Counter()
            for example in result.examples:
                for label in example.labels:
                    if label != POSITIVE_LABEL:
                        counts[label] += 1

            if not counts:
                return

            result.rows = [
                ResultRow(data={"label": label, "review_count": count, "source": "rag_candidates"})
                for label, count in counts.most_common(query.limit)
            ]
            result.raw["rag_aggregate_note"] = (
                "Rows aggregated from Qdrant candidates after semantic search/validation."
            )
            result.raw.setdefault("trace_steps", []).append(
                TraceStep(
                    id="rag_aggregate",
                    title="Агрегация найденных отзывов",
                    duration_ms=0,
                    input={"examples_count": len(result.examples)},
                    output={"rows_count": len(result.rows), "top_rows": [row.data for row in result.rows[:5]]},
                ).model_dump()
            )

    def _add_tool_step(
        self,
        merged: StructuredResult,
        tool: ToolName,
        query: ParsedQuery,
        result: StructuredResult,
        started_at: float,
    ) -> None:
        merged.raw.setdefault("trace_steps", []).append(
            TraceStep(
                id=f"tool_{tool.value}",
                title=f"Запуск инструмента: {tool.value}",
                status="warning" if result.warnings else "ok",
                duration_ms=int((perf_counter() - started_at) * 1000),
                input={
                    "intent": query.intent,
                    "filters": query.filters.model_dump(),
                    "semantic_query": query.semantic_query,
                    "limit": query.limit,
                },
                output={
                    "metrics_count": len(result.metrics),
                    "rows_count": len(result.rows),
                    "examples_count": len(result.examples),
                    "warnings": result.warnings,
                    "sample_rows": [row.data for row in result.rows[:3]],
                    "sample_examples": [example.model_dump() for example in result.examples[:3]],
                },
            ).model_dump()
        )
