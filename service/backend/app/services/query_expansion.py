from __future__ import annotations


EXPANSION_PATTERNS: dict[str, list[str]] = {
    "хрупк": ["сломался", "разбитый", "треснул", "поврежденный", "раскололся"],
    "разбит": ["сломался", "треснул", "поврежденный", "хрупкий"],
    "слом": ["разбитый", "треснул", "не работает", "поврежденный"],
    "маломер": ["маленький", "тесный", "не подошел размер", "узкий"],
    "большемер": ["большой", "широкий", "не подошел размер", "свободный"],
    "размер": ["маломерит", "большемерит", "тесный", "не подошел"],
    "упаков": ["коробка", "пакет", "помятая", "вскрытая", "без упаковки"],
    "комплект": ["не хватает", "нет детали", "пришел некомплект", "без инструкции"],
    "возврат": ["вернуть", "деньги", "отказ", "возврат средств"],
    "достав": ["получение", "пункт выдачи", "курьер", "задержка", "пришел поврежденный"],
    "запах": ["воняет", "резкий запах", "химический запах"],
    "цвет": ["оттенок", "не тот цвет", "отличается от фото"],
    "облож": ["книга", "страницы", "помятая", "порванная", "поврежденная"],
    "книг": ["обложка", "страницы", "учебник", "печатное издание"],
}


def expand_semantic_query(query: str | None) -> str | None:
    if not query:
        return query

    lowered = query.lower()
    additions: list[str] = []
    for stem, variants in EXPANSION_PATTERNS.items():
        if stem in lowered:
            additions.extend(item for item in variants if item not in additions)

    if not additions:
        return query
    return f"{query}. Похожие формулировки: {', '.join(additions)}"


def query_terms(query: str | None) -> list[str]:
    if not query:
        return []
    normalized = "".join(char.lower() if char.isalnum() else " " for char in query)
    terms = [term for term in normalized.split() if len(term) >= 4]
    expanded = expand_semantic_query(query) or query
    normalized_expanded = "".join(char.lower() if char.isalnum() else " " for char in expanded)
    for term in normalized_expanded.split():
        if len(term) >= 4 and term not in terms:
            terms.append(term)
    return terms
