from dataclasses import dataclass
from app.schemas.query import AnswerMode, GroupBy, Intent, ToolName


@dataclass(frozen=True)
class TemplateConfig:
    id: str
    title: str
    description: str
    intent: Intent
    default_tools: list[ToolName]
    default_answer_mode: AnswerMode
    allow_llm_summary: bool
    default_group_by: GroupBy | None = None
    needs_semantic_query: bool = False
    label_mode: str = "optional"
    keyword_mode: str = "optional"


# Только шаблоны, которые имеют смысл при текущих данных:
# review_id, review_date, text, labels.
TEMPLATE_REGISTRY: dict[str, TemplateConfig] = {
    "count_by_problem": TemplateConfig(
        id="count_by_problem",
        title="Количество отзывов по проблеме",
        description="Считает, сколько отзывов относится к выбранной проблеме.",
        intent=Intent.COUNT_BY_PROBLEM,
        default_tools=[ToolName.POSTGRES],
        default_answer_mode=AnswerMode.TEMPLATE,
        allow_llm_summary=True,
        label_mode="required",
        keyword_mode="hidden",
    ),
    "top_problems": TemplateConfig(
        id="top_problems",
        title="Топ проблем",
        description="Показывает самые частые проблемы за выбранный период.",
        intent=Intent.TOP_PROBLEMS,
        default_tools=[ToolName.POSTGRES],
        default_answer_mode=AnswerMode.TEMPLATE,
        allow_llm_summary=True,
        default_group_by=GroupBy.LABEL,
        label_mode="subset",
        keyword_mode="hidden",
    ),
    "problem_dynamics": TemplateConfig(
        id="problem_dynamics",
        title="Динамика проблемы",
        description="Показывает динамику отзывов по проблемам по дням, неделям или месяцам.",
        intent=Intent.PROBLEM_DYNAMICS,
        default_tools=[ToolName.POSTGRES],
        default_answer_mode=AnswerMode.TEMPLATE,
        allow_llm_summary=True,
        default_group_by=GroupBy.WEEK,
        label_mode="subset",
        keyword_mode="hidden",
    ),
    "review_samples": TemplateConfig(
        id="review_samples",
        title="Примеры отзывов",
        description="Показывает конкретные отзывы по проблеме, периоду или ключевому слову.",
        intent=Intent.REVIEW_SAMPLES,
        default_tools=[ToolName.POSTGRES],
        default_answer_mode=AnswerMode.TEMPLATE,
        allow_llm_summary=True,
        label_mode="optional",
        keyword_mode="optional",
    ),
    "review_examples": TemplateConfig(
        id="review_examples",
        title="RAG: похожие отзывы",
        description="Ищет похожие отзывы по смыслу через Qdrant и возвращает примеры для объяснения.",
        intent=Intent.REVIEW_EXAMPLES,
        default_tools=[ToolName.QDRANT],
        default_answer_mode=AnswerMode.LLM,
        allow_llm_summary=True,
        needs_semantic_query=True,
        label_mode="optional",
        keyword_mode="required",
    ),
    "top_products_by_problem": TemplateConfig(
        id="top_products_by_problem",
        title="Топ товаров по проблеме",
        description="Показывает товары, у которых больше всего отзывов по выбранной проблеме или группе проблем.",
        intent=Intent.TOP_PRODUCTS_BY_PROBLEM,
        default_tools=[ToolName.POSTGRES],
        default_answer_mode=AnswerMode.TEMPLATE,
        allow_llm_summary=True,
        default_group_by=GroupBy.PRODUCT,
        label_mode="optional",
        keyword_mode="hidden",
    ),
    "period_comparison": TemplateConfig(
        id="period_comparison",
        title="Сравнение периодов",
        description="Сравнивает текущий период с предыдущим равным периодом.",
        intent=Intent.PERIOD_COMPARISON,
        default_tools=[ToolName.POSTGRES],
        default_answer_mode=AnswerMode.TEMPLATE,
        allow_llm_summary=True,
        default_group_by=GroupBy.LABEL,
        label_mode="subset",
        keyword_mode="optional",
    ),
    "problem_share": TemplateConfig(
        id="problem_share",
        title="Доли проблем",
        description="Показывает долю каждой проблемы среди всех отзывов за период.",
        intent=Intent.PROBLEM_SHARE,
        default_tools=[ToolName.POSTGRES],
        default_answer_mode=AnswerMode.TEMPLATE,
        allow_llm_summary=True,
        default_group_by=GroupBy.LABEL,
        label_mode="subset",
        keyword_mode="hidden",
    ),
    "problem_growth": TemplateConfig(
        id="problem_growth",
        title="Рост проблем",
        description="Показывает проблемы, которые сильнее всего выросли относительно предыдущего периода.",
        intent=Intent.PROBLEM_GROWTH,
        default_tools=[ToolName.POSTGRES],
        default_answer_mode=AnswerMode.TEMPLATE,
        allow_llm_summary=True,
        default_group_by=GroupBy.LABEL,
        label_mode="subset",
        keyword_mode="optional",
    ),
    "label_cooccurrence": TemplateConfig(
        id="label_cooccurrence",
        title="Совместные проблемы",
        description="Показывает, какие проблемы чаще всего встречаются вместе в одном отзыве.",
        intent=Intent.LABEL_COOCCURRENCE,
        default_tools=[ToolName.POSTGRES],
        default_answer_mode=AnswerMode.TEMPLATE,
        allow_llm_summary=True,
        label_mode="subset",
        keyword_mode="hidden",
    ),
    "keyword_search": TemplateConfig(
        id="keyword_search",
        title="Поиск по слову",
        description="Ищет отзывы по слову или фразе через PostgreSQL ILIKE.",
        intent=Intent.KEYWORD_SEARCH,
        default_tools=[ToolName.POSTGRES],
        default_answer_mode=AnswerMode.TEMPLATE,
        allow_llm_summary=True,
        label_mode="optional",
        keyword_mode="required",
    ),
    "positive_vs_problem": TemplateConfig(
        id="positive_vs_problem",
        title="Положительные против проблемных",
        description="Показывает долю положительных/нейтральных и проблемных отзывов во времени.",
        intent=Intent.POSITIVE_VS_PROBLEM,
        default_tools=[ToolName.POSTGRES],
        default_answer_mode=AnswerMode.TEMPLATE,
        allow_llm_summary=True,
        default_group_by=GroupBy.MONTH,
        label_mode="hidden",
        keyword_mode="hidden",
    ),
}


def list_templates() -> list[dict]:
    return [
        {
            "id": template.id,
            "title": template.title,
            "description": template.description,
            "default_answer_mode": template.default_answer_mode,
            "allow_llm_summary": template.allow_llm_summary,
            "needs_semantic_query": template.needs_semantic_query,
            "label_mode": template.label_mode,
            "keyword_mode": template.keyword_mode,
            "default_group_by": template.default_group_by,
        }
        for template in TEMPLATE_REGISTRY.values()
    ]


def get_template(template_id: str) -> TemplateConfig:
    try:
        return TEMPLATE_REGISTRY[template_id]
    except KeyError as exc:
        raise ValueError(f"Unknown template_id: {template_id}") from exc
