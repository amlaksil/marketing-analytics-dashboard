CREATE TABLE telegram_post_performance (
    id SERIAL PRIMARY KEY,
    post_id BIGINT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
		url VARCHAR(250),
    views BIGINT,
    UNIQUE(post_id, timestamp)
);

CREATE INDEX idx_post_id ON telegram_post_performance (post_id);
CREATE INDEX idx_timestamp ON telegram_post_performance(timestamp);
