DROP TABLE memoria;

CREATE TABLE memoria(
    id UUID primary key,
    embedding vector(768),
    metadados jsonb
);