import json
import ast
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import DatetimeRange, Filter, FieldCondition, MatchAny, MatchText, MatchValue, Range

from app.core.config import get_settings
from app.schemas.query import ParsedQuery, ReviewExample, StructuredResult
from app.services.embedding_service import EmbeddingTimeoutError, embedding_service
from app.services.query_expansion import expand_semantic_query, query_terms


class QdrantTool:
    """Инструмент для semantic search по отзывам.

    В MVP метод `_embed` оставлен как точка расширения.
    Его можно заменить на локальную embedding-модель или внешний embedding API.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def run(self, query: ParsedQuery) -> StructuredResult:
        result = StructuredResult(parsed_query=query)
        debug: dict[str, Any] = {
            "candidate_count": 0,
            "returned_count": 0,
            "embedding_ms": 0,
            "qdrant_ms": 0,
            "keyword_rescue_ms": 0,
            "rerank_ms": 0,
            "validation_ms": 0,
        }
        result.raw["qdrant_debug"] = debug

        if not self.settings.qdrant_url:
            result.warnings.append("QDRANT_URL не задан. Qdrant-инструмент не был выполнен.")
            return result

        if not query.semantic_query:
            result.warnings.append("semantic_query пустой. Для Qdrant-поиска нужен текстовый запрос.")
            return result

        try:
            client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key or None,
                timeout=self.settings.rag_qdrant_timeout_seconds,
            )
            expanded_query = expand_semantic_query(query.semantic_query) or query.semantic_query
            result.raw["expanded_semantic_query"] = expanded_query
            vector, embedding_debug = self._embed(expanded_query)
            debug.update(embedding_debug)
            qdrant_filter = self._build_filter(query)

            fetch_limit = min(max(query.limit * 5, query.limit), self.settings.rag_top_k)
            qdrant_started_at = perf_counter()
            hits = client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=vector,
                query_filter=qdrant_filter,
                limit=fetch_limit,
                with_payload=True,
            )
            debug["qdrant_ms"] = int((perf_counter() - qdrant_started_at) * 1000)

            rescue_started_at = perf_counter()
            hits = [
                *hits,
                *self._keyword_rescue_hits(
                    client=client,
                    query=query,
                    qdrant_filter=qdrant_filter,
                    existing_ids={str((hit.payload or {}).get("review_id") or hit.id) for hit in hits},
                ),
            ]
            debug["keyword_rescue_ms"] = int((perf_counter() - rescue_started_at) * 1000)
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, EmbeddingTimeoutError):
                result.warnings.append(str(exc))
            else:
                result.warnings.append(f"Ошибка Qdrant-инструмента: {exc}")
            return result

        debug["candidate_count"] = len(hits)
        rerank_started_at = perf_counter()
        ranked_hits = self._rerank_hits(hits, query)
        debug["rerank_ms"] = int((perf_counter() - rerank_started_at) * 1000)
        selected_hits = self._diversify_hits(ranked_hits, query.limit)
        validation_started_at = perf_counter()
        selected_hits = self._validate_hits_with_llm(query.semantic_query, selected_hits)
        debug["validation_ms"] = int((perf_counter() - validation_started_at) * 1000)
        debug["returned_count"] = len(selected_hits[: query.limit])

        for hit in selected_hits[: query.limit]:
            payload = hit.payload or {}
            labels = self._payload_labels(payload)
            text = self._payload_text(payload)
            result.examples.append(
                ReviewExample(
                    review_id=str(payload.get("review_id") or hit.id),
                    text=str(text),
                    labels=list(labels),
                    product_id=str(payload.get("product_id")) if payload.get("product_id") is not None else None,
                    product_name=payload.get("product_name"),
                    category=payload.get("category"),
                    brand=payload.get("brand"),
                    rating=payload.get("rating"),
                    date=str(payload.get("review_date")) if payload.get("review_date") else None,
                    score=float(hit.score) if hit.score is not None else None,
                )
            )

        return result

    def _embed(self, text: str) -> tuple[list[float], dict[str, Any]]:
        """Строит embedding для semantic search.

        Для корректного поиска модель должна совпадать с моделью, которой строилась
        Qdrant collection. Если коллекция уже построена BAAI/bge-m3, нужно либо
        подключить такой же encoder, либо перестроить коллекцию под OpenAI embedding.
        """
        return embedding_service.embed(text)

    def _build_filter(self, query: ParsedQuery) -> Filter | None:
        conditions = []
        f = query.filters

        if f.labels:
            conditions.append(FieldCondition(key="predicted_labels", match=MatchAny(any=f.labels)))
        if f.date_from or f.date_to:
            conditions.append(
                FieldCondition(
                    key="review_date",
                    range=DatetimeRange(
                        gte=f"{f.date_from}T00:00:00Z" if f.date_from else None,
                        lte=f"{f.date_to}T23:59:59Z" if f.date_to else None,
                    ),
                )
            )
        if f.category:
            conditions.append(FieldCondition(key="category", match=MatchValue(value=f.category)))
        if f.brand:
            conditions.append(FieldCondition(key="brand", match=MatchValue(value=f.brand)))
        if f.product_id:
            conditions.append(FieldCondition(key="product_id", match=MatchValue(value=f.product_id)))
        if f.product_name:
            conditions.append(FieldCondition(key="product_name", match=MatchText(text=f.product_name)))
        if f.min_rating is not None or f.max_rating is not None:
            conditions.append(
                FieldCondition(
                    key="rating",
                    range=Range(gte=f.min_rating, lte=f.max_rating),
                )
            )

        if not conditions:
            return None
        return Filter(must=conditions)

    def _payload_text(self, payload: dict[str, Any]) -> str:
        return str(
            payload.get("text")
            or payload.get("review_text")
            or payload.get("review_text_preview")
            or ""
        )

    def _payload_labels(self, payload: dict[str, Any]) -> list[str]:
        labels = self._parse_labels(payload.get("labels"))
        if labels:
            return labels
        labels = self._parse_labels(payload.get("predicted_labels"))
        if labels:
            return labels
        return self._parse_labels(payload.get("predicted_labels_str"))

    def _parse_labels(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, tuple):
            return [str(item).strip() for item in value if str(item).strip()]

        raw = str(value).strip()
        if not raw or raw.lower() in {"nan", "none", "null", "[]"}:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(raw)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except Exception:
                    pass

        for separator in ("|", ";", ","):
            if separator in raw:
                return [item.strip() for item in raw.split(separator) if item.strip()]

        return [raw]

    def _diversify_hits(self, hits: list, limit: int) -> list:
        """Разбавляет выдачу по label/product, чтобы LLM видел разные типы примеров."""
        buckets: dict[str, list] = {}
        for hit in hits:
            payload = hit.payload or {}
            labels = payload.get("labels") or []
            if not labels:
                labels = self._payload_labels(payload)
            first_label = labels[0] if labels else "no_label"
            product = payload.get("product_id") or payload.get("product_name") or "no_product"
            key = f"{first_label}|{product}"
            buckets.setdefault(key, []).append(hit)

        selected = []
        while len(selected) < limit and buckets:
            for key in list(buckets.keys()):
                bucket = buckets[key]
                if bucket:
                    selected.append(bucket.pop(0))
                    if len(selected) >= limit:
                        break
                if not bucket:
                    del buckets[key]

        return selected

    def _rerank_hits(self, hits: list, query: ParsedQuery) -> list:
        terms = query_terms(query.semantic_query)
        damage_query = bool(query.semantic_query and any(stem in query.semantic_query.lower() for stem in ("хруп", "разбит", "трес", "слом", "поврежд")))
        damage_stems = ("разбит", "слом", "трес", "поврежд", "раскол", "развал", "лопнул")

        def score(hit: Any) -> float:
            payload = hit.payload or {}
            labels = self._payload_labels(payload)
            text = self._payload_text(payload).lower()
            product_name = str(payload.get("product_name") or "").lower()
            category = str(payload.get("category") or "").lower()
            brand = str(payload.get("brand") or "").lower()

            value = float(hit.score or 0.0)
            if query.filters.labels and any(label in labels for label in query.filters.labels):
                value += 0.25
            if query.filters.product_id and str(payload.get("product_id")) == query.filters.product_id:
                value += 0.2
            if query.filters.product_name and query.filters.product_name.lower() in product_name:
                value += 0.18
            if query.filters.category and query.filters.category.lower() == category:
                value += 0.12
            if query.filters.brand and query.filters.brand.lower() == brand:
                value += 0.12

            rating = self._to_int(payload.get("rating"))
            if rating is not None:
                if query.filters.min_rating is not None and rating >= query.filters.min_rating:
                    value += 0.05
                if query.filters.max_rating is not None and rating <= query.filters.max_rating:
                    value += 0.05

            term_hits = sum(
                1
                for term in terms
                if self._term_matches(term, text) or self._term_matches(term, product_name) or self._term_matches(term, category)
            )
            value += min(term_hits, 6) * 0.035
            if damage_query:
                damage_hits = sum(1 for stem in damage_stems if stem in text)
                value += min(damage_hits, 4) * 0.18

            text_len = len(text)
            if 40 <= text_len <= 700:
                value += 0.08
            elif text_len < 12:
                value -= 0.08
            return value

        return sorted(hits, key=score, reverse=True)

    def _keyword_rescue_hits(
        self,
        *,
        client: QdrantClient,
        query: ParsedQuery,
        qdrant_filter: Filter | None,
        existing_ids: set[str],
    ) -> list:
        terms = query_terms(query.semantic_query)
        if not terms:
            return []

        damage_query = bool(query.semantic_query and any(stem in query.semantic_query.lower() for stem in ("хруп", "разбит", "трес", "слом", "поврежд")))
        rescue_stems = ["разбит", "слом", "трес", "поврежд", "раскол", "развал", "лопнул"] if damage_query else []
        rescue_stems.extend(term for term in terms if term not in rescue_stems)

        rescued = []
        offset = None
        scanned = 0
        max_scan = self.settings.rag_keyword_rescue_max_scan
        rescue_limit = self.settings.rag_keyword_rescue_limit
        while scanned < max_scan and len(rescued) < rescue_limit:
            points, offset = client.scroll(
                collection_name=self.settings.qdrant_collection,
                scroll_filter=qdrant_filter,
                limit=min(300, max_scan),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            scanned += len(points)
            for point in points:
                payload = point.payload or {}
                review_id = str(payload.get("review_id") or point.id)
                if review_id in existing_ids:
                    continue
                text = self._payload_text(payload).lower()
                product_name = str(payload.get("product_name") or "").lower()
                if any(stem in text or stem in product_name for stem in rescue_stems):
                    rescued.append(SimpleNamespace(id=point.id, payload=payload, score=1.0))
                    existing_ids.add(review_id)
                    if len(rescued) >= rescue_limit:
                        break
            if offset is None:
                break
        return rescued

    def _term_matches(self, term: str, text: str) -> bool:
        if term in text:
            return True
        if len(term) >= 6 and term[:5] in text:
            return True
        return False

    def _to_int(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _validate_hits_with_llm(self, semantic_query: str | None, hits: list) -> list:
        if not semantic_query or not hits or not self.settings.rag_validate_candidates:
            return hits
        if not self.settings.openai_api_key:
            return hits

        candidates = []
        for hit in hits[: self.settings.rag_validation_limit]:
            payload = hit.payload or {}
            candidates.append(
                {
                    "id": str(payload.get("review_id") or hit.id),
                    "text": self._payload_text(payload)[:900],
                    "labels": self._payload_labels(payload),
                    "product_name": payload.get("product_name"),
                    "category": payload.get("category"),
                }
            )

        prompt = f"""
Ты валидатор RAG-кандидатов для сервиса анализа отзывов.
Нужно оставить только отзывы, которые действительно релевантны вопросу пользователя.

Вопрос:
{semantic_query}

Кандидаты:
{json.dumps(candidates, ensure_ascii=False)}

Верни только JSON:
{{"relevant_ids": ["id1", "id2"]}}

Правила:
- если пользователь спрашивает про книги, обложки, страницы или учебники, оставляй отзывы, где это явно или по смыслу относится к книгам;
- если отзыв нерелевантен вопросу, исключи его;
- не добавляй id, которых нет в кандидатах.
""".strip()

        try:
            client = OpenAI(api_key=self.settings.openai_api_key, timeout=self.settings.rag_validation_timeout_seconds)
            response = client.responses.create(
                model=self.settings.openai_model,
                input=prompt,
            )
            relevant_ids = set(json.loads(response.output_text).get("relevant_ids", []))
        except Exception:  # noqa: BLE001
            return hits

        if not relevant_ids:
            return []

        return [
            hit
            for hit in hits
            if str((hit.payload or {}).get("review_id") or hit.id) in relevant_ids
        ]
