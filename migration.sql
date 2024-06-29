-- Create extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table
CREATE TABLE jobs_embeddings (
    id SERIAL PRIMARY KEY,
    embeddings VECTOR(384)
    job_id INTEGER NOT NULL UNIQUE,
);

-- Create index
CREATE INDEX ON jobs_embeddings USING ivfflat (embeddings);
