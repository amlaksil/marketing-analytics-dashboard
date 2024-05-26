CREATE TABLE app_download_data (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    installs VARCHAR(50),
    min_installs BIGINT,
    real_installs BIGINT,
    score FLOAT CHECK (score >= 0 AND score <= 5),
    ratings BIGINT,
    reviews BIGINT,
    developer VARCHAR(255),
    developer_id VARCHAR(255),
    developer_email VARCHAR(255),
    developer_website VARCHAR(255),
    developer_address TEXT,
    released DATE,
    last_updated_on DATE,
    updated BIGINT,
    version VARCHAR(50),
    app_id VARCHAR(255) NOT NULL,
    url VARCHAR(255)
);

CREATE INDEX idx_updated ON app_download_data (updated);
CREATE INDEX idx_app_id ON app_download_data (app_id);
