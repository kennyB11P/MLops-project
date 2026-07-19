import os
import re
from typing import Any
from app.domain.labels import POSITIVE_LABEL
from app.schemas.query import GroupBy, Intent, ParsedQuery


class SQLBuilder:
    """Собирает безопасные SQL-шаблоны под основные intent.

    Все пользовательские значения передаются через params.
    Динамически подставляется только заранее выбранная гранулярность даты.
    """

    PROBLEM_ONLY_INTENTS = {
        Intent.COUNT_BY_PROBLEM,
        Intent.TOP_PROBLEMS,
        Intent.PROBLEM_DYNAMICS,
        Intent.PERIOD_COMPARISON,
        Intent.PROBLEM_SHARE,
        Intent.PROBLEM_GROWTH,
        Intent.LABEL_COOCCURRENCE,
        Intent.TOP_PRODUCTS_BY_PROBLEM,
        Intent.PROBLEM_GROWTH_ANALYSIS,
    }

    def build(self, query: ParsedQuery) -> tuple[str, dict[str, Any]]:
        exclude_positive_label = self._should_exclude_positive_label(query)
        where_sql, params = self._build_filters(query, exclude_positive_label=exclude_positive_label)

        if query.intent == Intent.COUNT_BY_PROBLEM:
            return self._count_by_problem(where_sql, params)
        if query.intent == Intent.TOP_PROBLEMS:
            return self._top_problems(where_sql, params, query.limit)
        if query.intent == Intent.PROBLEM_DYNAMICS:
            return self._problem_dynamics(where_sql, params, query.group_by or GroupBy.WEEK)
        if query.intent in {Intent.REVIEW_SAMPLES, Intent.REVIEW_EXAMPLES, Intent.KEYWORD_SEARCH}:
            return self._review_samples(query)
        if query.intent == Intent.PERIOD_COMPARISON:
            return self._period_comparison(query, order_by="abs_delta")
        if query.intent in {Intent.PROBLEM_GROWTH, Intent.PROBLEM_GROWTH_ANALYSIS}:
            return self._period_comparison(query, order_by="growth")
        if query.intent == Intent.PROBLEM_SHARE:
            return self._problem_share(query)
        if query.intent == Intent.LABEL_COOCCURRENCE:
            return self._label_cooccurrence(query)
        if query.intent == Intent.POSITIVE_VS_PROBLEM:
            return self._positive_vs_problem(query, query.group_by or GroupBy.MONTH)
        if query.intent == Intent.TOP_PRODUCTS_BY_PROBLEM:
            return self._top_products(query)

        return self._top_problems(where_sql, params, query.limit)

    def _build_filters(
        self,
        query: ParsedQuery,
        *,
        include_labels: bool = True,
        include_dates: bool = True,
        include_keyword: bool = True,
        table_alias: str = "r",
        label_alias: str = "rl",
        exclude_positive_label: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        filters = query.filters
        clauses = ["1 = 1"]
        params: dict[str, Any] = {}

        if include_dates and filters.date_from:
            clauses.append(f"{table_alias}.review_date >= %(date_from)s")
            params["date_from"] = filters.date_from
        if include_dates and filters.date_to:
            clauses.append(f"{table_alias}.review_date <= %(date_to)s")
            params["date_to"] = filters.date_to
        if include_keyword and filters.keyword:
            clauses.append(f"{table_alias}.text ILIKE %(keyword)s")
            params["keyword"] = f"%{filters.keyword}%"
        if filters.category:
            clauses.append(f"{table_alias}.category = %(category)s")
            params["category"] = filters.category
        if filters.brand:
            clauses.append(f"{table_alias}.brand = %(brand)s")
            params["brand"] = filters.brand
        if filters.product_id:
            clauses.append(f"{table_alias}.product_id = %(product_id)s")
            params["product_id"] = filters.product_id
        if filters.product_name:
            clauses.append(f"{table_alias}.product_name ILIKE %(product_name)s")
            params["product_name"] = f"%{filters.product_name}%"
        if filters.min_rating is not None:
            clauses.append(f"{table_alias}.rating >= %(min_rating)s")
            params["min_rating"] = filters.min_rating
        if filters.max_rating is not None:
            clauses.append(f"{table_alias}.rating <= %(max_rating)s")
            params["max_rating"] = filters.max_rating
        if include_labels and filters.labels:
            clauses.append(f"{label_alias}.label = ANY(%(labels)s)")
            params["labels"] = filters.labels
        elif include_labels and exclude_positive_label:
            clauses.append(f"{label_alias}.label <> %(positive_label)s")
            params["positive_label"] = POSITIVE_LABEL

        return " AND ".join(clauses), params

    def _count_by_problem(self, where_sql: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        sql = f"""
        SELECT COUNT(DISTINCT r.review_id) AS review_count
        FROM reviews r
        LEFT JOIN review_labels rl ON rl.review_id = r.review_id
        WHERE {where_sql};
        """
        return sql, params

    def _top_problems(self, where_sql: str, params: dict[str, Any], limit: int) -> tuple[str, dict[str, Any]]:
        params = {**params, "limit": limit}
        sql = f"""
        SELECT rl.label, COUNT(DISTINCT r.review_id) AS review_count
        FROM reviews r
        JOIN review_labels rl ON rl.review_id = r.review_id
        WHERE {where_sql}
        GROUP BY rl.label
        ORDER BY review_count DESC
        LIMIT %(limit)s;
        """
        return sql, params

    def _problem_dynamics(self, where_sql: str, params: dict[str, Any], group_by: GroupBy) -> tuple[str, dict[str, Any]]:
        date_granularity = self._date_granularity(group_by)

        sql = f"""
        SELECT
            DATE_TRUNC('{date_granularity}', r.review_date)::date AS period,
            COALESCE(rl.label, 'Без класса') AS label,
            COUNT(DISTINCT r.review_id) AS review_count
        FROM reviews r
        LEFT JOIN review_labels rl ON rl.review_id = r.review_id
        WHERE {where_sql}
        GROUP BY period, label
        ORDER BY period ASC, review_count DESC;
        """
        return sql, params

    def _review_samples(self, query: ParsedQuery) -> tuple[str, dict[str, Any]]:
        search_text = query.filters.keyword
        where_sql, params = self._build_filters(query, include_labels=False, include_keyword=False)
        params = {**params, "limit": query.limit}

        label_filter_sql = ""
        if query.filters.labels:
            label_filter_sql = """
            AND EXISTS (
                SELECT 1
                FROM review_labels filter_rl
                WHERE filter_rl.review_id = r.review_id
                  AND filter_rl.label = ANY(%(labels)s)
            )
            """
            params["labels"] = query.filters.labels

        search_filter_sql = ""
        rank_select_sql = "0::real AS search_rank,"
        order_sql = "r.review_date DESC NULLS LAST, r.review_id"
        if search_text:
            tokens = self._search_tokens(search_text)
            params["search_text"] = search_text
            params["search_like"] = f"%{search_text}%"
            search_filter_sql = """
            AND (
                to_tsvector('russian', COALESCE(r.text, '') || ' ' || COALESCE(r.product_name, '') || ' ' || COALESCE(r.category, ''))
                    @@ plainto_tsquery('russian', %(search_text)s)
                OR r.text ILIKE %(search_like)s
                OR r.product_name ILIKE %(search_like)s
            """
            token_clauses = []
            for index, token in enumerate(tokens):
                params[f"search_token_{index}"] = f"%{token}%"
                token_clauses.append(
                    f"(r.text ILIKE %(search_token_{index})s OR r.product_name ILIKE %(search_token_{index})s)"
                )
            if token_clauses:
                search_filter_sql += "\n                OR (" + " AND ".join(token_clauses) + ")"
            search_filter_sql += ")"
            rank_select_sql = """
            (
                ts_rank_cd(
                    to_tsvector('russian', COALESCE(r.text, '') || ' ' || COALESCE(r.product_name, '') || ' ' || COALESCE(r.category, '')),
                    plainto_tsquery('russian', %(search_text)s)
                )
                + CASE WHEN r.text ILIKE %(search_like)s THEN 2 ELSE 0 END
            )::real AS search_rank,
            """
            order_sql = "search_rank DESC, r.review_date DESC NULLS LAST, r.review_id"

        sql = f"""
        SELECT
            {rank_select_sql}
            r.review_id,
            r.review_date::text AS date,
            r.product_id,
            r.product_name,
            r.brand,
            r.category,
            r.rating,
            r.text,
            ARRAY_REMOVE(ARRAY_AGG(DISTINCT all_rl.label ORDER BY all_rl.label), NULL) AS labels
        FROM reviews r
        LEFT JOIN review_labels all_rl ON all_rl.review_id = r.review_id
        WHERE {where_sql}
        {label_filter_sql}
        {search_filter_sql}
        GROUP BY r.review_id, r.review_date, r.product_id, r.product_name, r.brand, r.category, r.rating, r.text
        ORDER BY {order_sql}
        LIMIT %(limit)s;
        """
        return sql, params

    def _search_tokens(self, text: str) -> list[str]:
        tokens = []
        for token in re.findall(r"[0-9a-zа-яё]{3,}", text.lower()):
            if token not in tokens:
                tokens.append(token)
        return tokens[:6]

    def _period_comparison(self, query: ParsedQuery, *, order_by: str) -> tuple[str, dict[str, Any]]:
        # Если date_from/date_to заданы, считаем их текущим периодом.
        # Предыдущий период берется равной длины прямо перед текущим.
        # Для полного календарного месяца сравниваем с предыдущим календарным месяцем.
        # Если даты не заданы, берем последний месяц данных и предыдущий месяц.
        exclude_positive_label = self._should_exclude_positive_label(query)
        base_where_sql, base_params = self._build_filters(query, include_labels=False, include_dates=False)
        params: dict[str, Any] = {
            **base_params,
            "date_from": query.filters.date_from,
            "date_to": query.filters.date_to,
            "limit": query.limit,
        }

        label_filter_sql = "WHERE label <> %(positive_label)s" if exclude_positive_label else ""
        if exclude_positive_label:
            params["positive_label"] = POSITIVE_LABEL
        if query.filters.labels:
            label_filter_sql = "WHERE label = ANY(%(labels)s)"
            params["labels"] = query.filters.labels

        order_sql = "ABS(COALESCE(delta_abs, 0)) DESC"
        if order_by == "growth":
            order_sql = "delta_pct DESC NULLS LAST, delta_abs DESC"

        sql = f"""
        WITH max_date AS (
            SELECT MAX(review_date)::date AS max_review_date FROM reviews
        ),
        bounds AS (
            SELECT
                COALESCE(
                    %(date_from)s::date,
                    DATE_TRUNC('month', max_review_date)::date
                ) AS p2_from,
                COALESCE(
                    %(date_to)s::date,
                    (DATE_TRUNC('month', max_review_date) + INTERVAL '1 month - 1 day')::date
                ) AS p2_to
            FROM max_date
        ),
        periods AS (
            SELECT
                CASE
                    WHEN p2_from = DATE_TRUNC('month', p2_from)::date
                     AND p2_to = (DATE_TRUNC('month', p2_from) + INTERVAL '1 month - 1 day')::date
                    THEN (p2_from - INTERVAL '1 month')::date
                    ELSE (p2_from - ((p2_to - p2_from + 1) * INTERVAL '1 day'))::date
                END AS p1_from,
                (p2_from - INTERVAL '1 day')::date AS p1_to,
                p2_from,
                p2_to
            FROM bounds
        ),
        label_counts AS (
            SELECT
                rl.label,
                COUNT(DISTINCT r.review_id) FILTER (
                    WHERE r.review_date BETWEEN periods.p1_from AND periods.p1_to
                ) AS count_period_1,
                COUNT(DISTINCT r.review_id) FILTER (
                    WHERE r.review_date BETWEEN periods.p2_from AND periods.p2_to
                ) AS count_period_2,
                MIN(periods.p1_from)::text AS period_1_from,
                MIN(periods.p1_to)::text AS period_1_to,
                MIN(periods.p2_from)::text AS period_2_from,
                MIN(periods.p2_to)::text AS period_2_to
            FROM reviews r
            JOIN review_labels rl ON rl.review_id = r.review_id
            CROSS JOIN periods
            WHERE r.review_date BETWEEN periods.p1_from AND periods.p2_to
              AND {base_where_sql}
            GROUP BY rl.label
        ),
        final AS (
            SELECT
                label,
                count_period_1,
                count_period_2,
                count_period_2 - count_period_1 AS delta_abs,
                CASE
                    WHEN count_period_1 = 0 THEN NULL
                    ELSE ROUND(((count_period_2 - count_period_1)::numeric / count_period_1) * 100, 2)
                END AS delta_pct,
                period_1_from,
                period_1_to,
                period_2_from,
                period_2_to
            FROM label_counts
        )
        SELECT *
        FROM final
        {label_filter_sql}
        ORDER BY {order_sql}
        LIMIT %(limit)s;
        """
        return sql, params

    def _problem_share(self, query: ParsedQuery) -> tuple[str, dict[str, Any]]:
        where_sql, params = self._build_filters(query, include_labels=False)
        params = {**params, "limit": query.limit}

        label_filter_sql = "AND rl.label <> %(positive_label)s" if self._should_exclude_positive_label(query) else ""
        if label_filter_sql:
            params["positive_label"] = POSITIVE_LABEL
        if query.filters.labels:
            label_filter_sql = "AND rl.label = ANY(%(labels)s)"
            params["labels"] = query.filters.labels

        sql = f"""
        WITH filtered_reviews AS (
            SELECT DISTINCT r.review_id
            FROM reviews r
            WHERE {where_sql}
        ),
        total AS (
            SELECT COUNT(*) AS total_reviews FROM filtered_reviews
        )
        SELECT
            rl.label,
            COUNT(DISTINCT fr.review_id) AS review_count,
            ROUND(COUNT(DISTINCT fr.review_id)::numeric / NULLIF(total.total_reviews, 0) * 100, 2) AS share_pct
        FROM filtered_reviews fr
        JOIN review_labels rl ON rl.review_id = fr.review_id
        CROSS JOIN total
        WHERE 1 = 1
        {label_filter_sql}
        GROUP BY rl.label, total.total_reviews
        ORDER BY review_count DESC
        LIMIT %(limit)s;
        """
        return sql, params

    def _label_cooccurrence(self, query: ParsedQuery) -> tuple[str, dict[str, Any]]:
        where_sql, params = self._build_filters(query, include_labels=False)
        params = {**params, "limit": query.limit}

        selected_label_sql = ""
        if self._should_exclude_positive_label(query):
            selected_label_sql = """
            AND rl1.label <> %(positive_label)s
            AND rl2.label <> %(positive_label)s
            """
            params["positive_label"] = POSITIVE_LABEL
        if query.filters.labels:
            selected_label_sql = "AND (rl1.label = ANY(%(labels)s) OR rl2.label = ANY(%(labels)s))"
            params["labels"] = query.filters.labels

        sql = f"""
        SELECT
            rl1.label AS label_1,
            rl2.label AS label_2,
            COUNT(DISTINCT r.review_id) AS review_count
        FROM reviews r
        JOIN review_labels rl1 ON rl1.review_id = r.review_id
        JOIN review_labels rl2 ON rl2.review_id = r.review_id AND rl1.label < rl2.label
        WHERE {where_sql}
        {selected_label_sql}
        GROUP BY rl1.label, rl2.label
        ORDER BY review_count DESC
        LIMIT %(limit)s;
        """
        return sql, params

    def _positive_vs_problem(self, query: ParsedQuery, group_by: GroupBy) -> tuple[str, dict[str, Any]]:
        where_sql, params = self._build_filters(query, include_labels=False)
        date_granularity = self._date_granularity(group_by)

        sql = f"""
        SELECT
            DATE_TRUNC('{date_granularity}', r.review_date)::date AS period,
            COUNT(DISTINCT r.review_id) FILTER (
                WHERE EXISTS (
                    SELECT 1 FROM review_labels rl
                    WHERE rl.review_id = r.review_id AND rl.label = %(positive_label)s
                )
            ) AS positive_count,
            COUNT(DISTINCT r.review_id) FILTER (
                WHERE EXISTS (
                    SELECT 1 FROM review_labels rl
                    WHERE rl.review_id = r.review_id AND rl.label <> %(positive_label)s
                )
            ) AS problem_count,
            ROUND(
                COUNT(DISTINCT r.review_id) FILTER (
                    WHERE EXISTS (
                        SELECT 1 FROM review_labels rl
                        WHERE rl.review_id = r.review_id AND rl.label <> %(positive_label)s
                    )
                )::numeric / NULLIF(COUNT(DISTINCT r.review_id), 0) * 100,
                2
            ) AS problem_share_pct
        FROM reviews r
        WHERE {where_sql}
        GROUP BY period
        ORDER BY period ASC;
        """
        return sql, {**params, "positive_label": POSITIVE_LABEL}

    def _top_products(self, query: ParsedQuery) -> tuple[str, dict[str, Any]]:
        total_where_sql, total_params = self._build_filters(query, include_labels=False)
        problem_where_sql, problem_params = self._build_filters(
            query,
            exclude_positive_label=self._should_exclude_positive_label(query),
        )
        params = {
            **total_params,
            **problem_params,
            "limit": query.limit,
            "min_total_reviews": int(os.getenv("MIN_TOTAL_REVIEWS_FOR_PRODUCT_RISK", "5")),
        }
        sql = f"""
        WITH product_totals AS (
            SELECT
                r.product_id,
                MAX(r.product_name) FILTER (WHERE r.product_name IS NOT NULL AND r.product_name <> '') AS product_name,
                MAX(r.category) FILTER (WHERE r.category IS NOT NULL AND r.category <> '') AS category,
                MAX(r.brand) FILTER (WHERE r.brand IS NOT NULL AND r.brand <> '') AS brand,
                COUNT(DISTINCT r.review_id) AS total_reviews,
                ROUND(AVG(r.rating)::numeric, 2) AS avg_rating,
                ROUND(
                    COUNT(DISTINCT r.review_id) FILTER (WHERE r.rating <= 2)::numeric
                    / NULLIF(COUNT(DISTINCT r.review_id) FILTER (WHERE r.rating IS NOT NULL), 0) * 100,
                    2
                ) AS negative_rating_share
            FROM reviews r
            WHERE {total_where_sql}
                AND COALESCE(r.product_id, '') NOT IN ('', '0')
            GROUP BY r.product_id
        ),
        product_problems AS (
            SELECT
                r.product_id,
                COUNT(DISTINCT r.review_id) AS problem_reviews,
                ARRAY_REMOVE(ARRAY_AGG(DISTINCT rl.label ORDER BY rl.label), NULL) AS problem_labels
            FROM reviews r
            JOIN review_labels rl ON rl.review_id = r.review_id
            WHERE {problem_where_sql}
                AND COALESCE(r.product_id, '') NOT IN ('', '0')
            GROUP BY r.product_id
        )
        SELECT
            pt.product_id,
            pt.product_name,
            pt.category,
            pt.brand,
            pp.problem_reviews,
            pt.total_reviews,
            ROUND(pp.problem_reviews::numeric / NULLIF(pt.total_reviews, 0) * 100, 2) AS problem_share_pct,
            pt.avg_rating,
            pt.negative_rating_share,
            pp.problem_labels,
            ROUND(
                (
                    (pp.problem_reviews::numeric / NULLIF(pt.total_reviews, 0) * 100)
                    * LN(1 + pt.total_reviews)
                )::numeric,
                2
            ) AS risk_score
        FROM product_problems pp
        JOIN product_totals pt ON pt.product_id = pp.product_id
        WHERE pt.total_reviews >= %(min_total_reviews)s
        ORDER BY pp.problem_reviews DESC, risk_score DESC, pt.total_reviews DESC
        LIMIT %(limit)s;
        """
        return sql, params

    def _date_granularity(self, group_by: GroupBy) -> str:
        return {
            GroupBy.DAY: "day",
            GroupBy.WEEK: "week",
            GroupBy.MONTH: "month",
        }.get(group_by, "week")

    def _should_exclude_positive_label(self, query: ParsedQuery) -> bool:
        return query.intent in self.PROBLEM_ONLY_INTENTS and not query.filters.labels
