from enum import StrEnum
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class QuerySource(StrEnum):
    TEMPLATE_UI = "template_ui"
    CHAT = "chat"


class ToolName(StrEnum):
    POSTGRES = "postgres"
    QDRANT = "qdrant"


class AnswerMode(StrEnum):
    TEMPLATE = "template"
    LLM = "llm"


class Intent(StrEnum):
    COUNT_BY_PROBLEM = "count_by_problem"
    TOP_PROBLEMS = "top_problems"
    PROBLEM_DYNAMICS = "problem_dynamics"
    REVIEW_SAMPLES = "review_samples"
    PERIOD_COMPARISON = "period_comparison"
    PROBLEM_SHARE = "problem_share"
    PROBLEM_GROWTH = "problem_growth"
    LABEL_COOCCURRENCE = "label_cooccurrence"
    KEYWORD_SEARCH = "keyword_search"
    POSITIVE_VS_PROBLEM = "positive_vs_problem"

    # Legacy intents. Оставлены, чтобы старые запросы не падали.
    TOP_PRODUCTS_BY_PROBLEM = "top_products_by_problem"
    REVIEW_EXAMPLES = "review_examples"
    PRODUCT_SUMMARY = "product_summary"
    RECOMMENDATIONS = "recommendations"
    PROBLEM_GROWTH_ANALYSIS = "problem_growth_analysis"


class GroupBy(StrEnum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    PRODUCT = "product"
    BRAND = "brand"
    CATEGORY = "category"
    LABEL = "label"


class ReviewFilters(BaseModel):
    date_from: str | None = Field(default=None, description="Дата начала в формате YYYY-MM-DD")
    date_to: str | None = Field(default=None, description="Дата конца в формате YYYY-MM-DD")
    labels: list[str] = Field(default_factory=list)
    keyword: str | None = Field(default=None, description="Слово или фраза для поиска по тексту отзыва")
    category: str | None = None
    brand: str | None = None
    product_id: str | None = None
    product_name: str | None = None
    min_rating: int | None = Field(default=None, ge=1, le=5)
    max_rating: int | None = Field(default=None, ge=1, le=5)

    @field_validator("date_from", "date_to", mode="before")
    @classmethod
    def normalize_date(cls, value: Any) -> str | None:
        if value is None:
            return None

        raw = str(value).strip()
        if not raw:
            return None

        try:
            return datetime.strptime(raw, "%Y-%m-%d").date().isoformat()
        except ValueError as exc:
            raise ValueError("Дата должна быть в формате YYYY-MM-DD") from exc

    @model_validator(mode="after")
    def validate_date_range(self) -> "ReviewFilters":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from не может быть позже date_to")
        return self


class ParsedQuery(BaseModel):
    source: QuerySource
    intent: Intent
    filters: ReviewFilters = Field(default_factory=ReviewFilters)
    group_by: GroupBy | None = None
    semantic_query: str | None = None
    tools: list[ToolName] = Field(default_factory=list)
    answer_mode: AnswerMode = AnswerMode.TEMPLATE
    limit: int = Field(default=20, ge=1, le=200)
    examples_limit: int = Field(default=5, ge=0, le=20)


class MetricBlock(BaseModel):
    name: str
    value: int | float | str | None
    unit: str | None = None


class ResultRow(BaseModel):
    data: dict[str, Any]


class ReviewExample(BaseModel):
    review_id: str | None = None
    text: str
    labels: list[str] = Field(default_factory=list)
    product_id: str | None = None
    product_name: str | None = None
    category: str | None = None
    brand: str | None = None
    rating: int | None = None
    date: str | None = None
    score: float | None = None


class StructuredResult(BaseModel):
    parsed_query: ParsedQuery
    metrics: list[MetricBlock] = Field(default_factory=list)
    rows: list[ResultRow] = Field(default_factory=list)
    examples: list[ReviewExample] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    id: str
    title: str
    status: str = "ok"
    duration_ms: int | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    parsed_query: ParsedQuery
    result: StructuredResult
    answer_mode: AnswerMode
    answer_text: str
    ui_blocks: list[dict[str, Any]] = Field(default_factory=list)
    execution_ms: int | None = None
    trace_steps: list[TraceStep] = Field(default_factory=list)


class TemplateExecuteRequest(BaseModel):
    filters: ReviewFilters = Field(default_factory=ReviewFilters)
    group_by: GroupBy | None = None
    semantic_query: str | None = None
    add_analytical_summary: bool = False
    limit: int = Field(default=20, ge=1, le=200)
    examples_limit: int = Field(default=5, ge=0, le=20)


class ChatAskRequest(BaseModel):
    message: str
    force_answer_mode: Literal["template", "llm"] | None = None
