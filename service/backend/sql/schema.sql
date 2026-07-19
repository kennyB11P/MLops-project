-- Минимальная схема под online-сервис.
-- Ее можно адаптировать под реальные колонки из датасета Wildberries.

CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    review_date DATE,
    text TEXT NOT NULL,
    rating INTEGER,
    product_id TEXT,
    product_name TEXT,
    brand TEXT,
    category TEXT,
    seller_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS review_labels (
    review_id TEXT NOT NULL REFERENCES reviews(review_id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    model_name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (review_id, label)
);

CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews(review_date);
CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(product_id);
CREATE INDEX IF NOT EXISTS idx_reviews_brand ON reviews(brand);
CREATE INDEX IF NOT EXISTS idx_reviews_category ON reviews(category);
CREATE INDEX IF NOT EXISTS idx_reviews_seller ON reviews(seller_id);
CREATE INDEX IF NOT EXISTS idx_reviews_fts_ru ON reviews USING GIN (
    to_tsvector('russian', COALESCE(text, '') || ' ' || COALESCE(product_name, '') || ' ' || COALESCE(category, ''))
);
CREATE INDEX IF NOT EXISTS idx_review_labels_label ON review_labels(label);

CREATE OR REPLACE VIEW review_primary_labels AS
WITH ranked AS (
    SELECT
        review_id,
        label,
        ROW_NUMBER() OVER (
            PARTITION BY review_id
            ORDER BY
                CASE
                    WHEN label = 'Проблема с качеством товара' THEN 1
                    WHEN label = 'Проблема с размером / посадкой' THEN 2
                    WHEN label = 'Проблема с комплектацией / упаковкой' THEN 3
                    WHEN label = 'Несоответствие карточке товара' THEN 4
                    WHEN label = 'Проблема с возвратом' THEN 5
                    WHEN label = 'Проблема доставки / получения' THEN 6
                    WHEN label = 'Цена / ценность' THEN 7
                    WHEN label = 'Другая проблема' THEN 8
                    WHEN label = 'Положительный / нейтральный отзыв' THEN 99
                    ELSE 100
                END,
                confidence DESC NULLS LAST,
                label
        ) AS rank
    FROM review_labels
)
SELECT review_id, label AS primary_label
FROM ranked
WHERE rank = 1;
