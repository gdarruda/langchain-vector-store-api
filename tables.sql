DROP TABLE memoria;

CREATE TABLE memoria(
    id UUID primary key,
    metadados jsonb,
    embedding vector(768)
);