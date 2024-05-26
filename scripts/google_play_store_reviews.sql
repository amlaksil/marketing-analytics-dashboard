CREATE TABLE google_play_store_reviews (
    id SERIAL PRIMARY KEY,
    review_id VARCHAR(255) NOT NULL,
		user_name VARCHAR(255) NOT NULL,
    user_image VARCHAR(255),
    content TEXT,
    score INT CHECK (score BETWEEN 1 AND 5),
    thumbs_up_count INT DEFAULT 0,
    review_created_version VARCHAR(50),
    review_at TIMESTAMP NOT NULL,
    reply_content TEXT,
    reply_at TIMESTAMP,
    app_version VARCHAR(50)
	);

CREATE INDEX idx_review_at ON google_play_store_reviews (review_at);
CREATE INDEX idx_score ON google_play_store_reviews (score);
CREATE INDEX idx_app_version ON google_play_store_reviews (app_version);
