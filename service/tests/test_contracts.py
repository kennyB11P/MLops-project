import pytest
from pydantic import ValidationError

from app.schemas.query import AnswerMode, ChatAskRequest, ReviewFilters, TemplateExecuteRequest
from app.services.chat_parser import ChatParser
from app.services.embedding_service import EmbeddingService
from app.services.template_parser import TemplateParser


def test_template_parser_count_by_problem() -> None:
    parser = TemplateParser()
    query = parser.parse(
        "count_by_problem",
        TemplateExecuteRequest(
            filters=ReviewFilters(labels=["Проблема доставки / получения"], category="Книги"),
            add_analytical_summary=True,
        ),
    )

    assert query.intent == "count_by_problem"
    assert query.filters.labels == ["Проблема доставки / получения"]
    assert query.filters.category == "Книги"
    assert query.answer_mode == AnswerMode.LLM
    assert query.tools == ["postgres"]


def test_template_parser_top_problems_keeps_empty_label_subset() -> None:
    parser = TemplateParser()
    query = parser.parse(
        "top_problems",
        TemplateExecuteRequest(filters=ReviewFilters(category="Книги")),
    )

    assert query.intent == "top_problems"
    assert query.filters.labels == []
    assert query.group_by == "label"


def test_template_parser_top_products_by_problem() -> None:
    parser = TemplateParser()
    query = parser.parse(
        "top_products_by_problem",
        TemplateExecuteRequest(
            filters=ReviewFilters(labels=["Проблема с качеством товара"]),
            limit=5,
        ),
    )

    assert query.intent == "top_products_by_problem"
    assert query.filters.labels == ["Проблема с качеством товара"]
    assert query.limit == 5


def test_review_filters_validate_date_format() -> None:
    filters = ReviewFilters(date_from="2025-09-01", date_to="")

    assert filters.date_from == "2025-09-01"
    assert filters.date_to is None

    with pytest.raises(ValidationError):
        ReviewFilters(date_from="01.09.2025")


def test_review_filters_reject_reversed_date_range() -> None:
    with pytest.raises(ValidationError):
        ReviewFilters(date_from="2025-10-01", date_to="2025-09-30")


def test_chat_rules_compare_october_with_september_uses_october_as_current_period() -> None:
    parser = ChatParser()

    query = parser._parse_with_rules(ChatAskRequest(message="сравни октябрь с сентябрем"))  # noqa: SLF001

    assert query.intent == "period_comparison"
    assert query.filters.date_from == "2025-10-01"
    assert query.filters.date_to == "2025-10-31"


def test_embedding_service_extracts_runpod_vector_shapes() -> None:
    service = EmbeddingService()

    assert service._extract_embedding_vector([0.1, 0.2]) == [0.1, 0.2]  # noqa: SLF001
    assert service._extract_embedding_vector({"output": {"embedding": [1, 2]}}) == [1.0, 2.0]  # noqa: SLF001
    assert service._extract_embedding_vector({"embeddings": [[3, 4]]}) == [3.0, 4.0]  # noqa: SLF001
