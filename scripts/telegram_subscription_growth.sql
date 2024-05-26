CREATE TABLE telegram_subscription_growth (
    id SERIAL PRIMARY KEY,
    post_id BIGINT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    url VARCHAR(250),
    views INT,
    UNIQUE(post_id, timestamp)
);

CREATE INDEX idx_post_id_boa ON telegram_subscription_growth(post_id);
CREATE INDEX idx_timestamp_boa ON telegram_subscription_growth(timestamp);
