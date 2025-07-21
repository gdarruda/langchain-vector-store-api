from uuid import UUID

from fastapi import FastAPI
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
from sqlalchemy import UUID, Column, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, declarative_base

app = FastAPI()
engine = create_engine(
    "postgresql+psycopg://langchain:langchain@localhost:6024/endpoint"
)
embedder = OllamaEmbeddings(model="nomic-embed-text:latest")


Base = declarative_base()


class Memoria(Base):
    __tablename__ = "memoria"
    id = Column(UUID, primary_key=True)
    embedding = Column(Vector(768))
    metadados = Column(JSONB)


class AddTextsPayload(BaseModel):
    texts: list[str]
    metadatas: list[dict]
    ids: list[str]


class AddEmbeddingsPayload(BaseModel):
    embeddings: list[list[float]]
    metadatas: list[dict]
    ids: list[str]


class SimilaritySearchByVectorPayload(BaseModel):
    embedding: list[float]
    k: int
    kwargs: dict


class SimilaritySearchResponse(BaseModel):
    id: str
    page_content: str
    metadata: dict


class GetByIdsResponse(BaseModel):
    id: str
    page_content: str
    metadata: dict


@app.delete("/delete")
def delete(ids: list[str]):

    deleted = False

    with Session(engine) as session:

        memories = session.query(Memoria).filter(Memoria.id.in_(ids))

        for memory in memories:
            session.delete(memory)
            deleted = True

        session.commit()

    return {"deleted": deleted}


@app.get("/get-by-ids")
def get_by_ids(ids: str) -> list[GetByIdsResponse]:
    splitted_ids = ids.split(",")

    with Session(engine) as session:
        memories = session.query(Memoria).filter(Memoria.id.in_(splitted_ids)).all()

    return [
        GetByIdsResponse(
            id=str(memory.id),
            metadata=memory.metadados,
            page_content=memory.metadados["data"],
        )
        for memory in memories
    ]


@app.post("/similarity-search-by-vector")
def similarity_search_by_vector(
    payload: SimilaritySearchByVectorPayload,
) -> list[SimilaritySearchResponse]:

    user_id = payload.kwargs["filter"]["user_id"]
    embedding = payload.embedding
    limit = payload.k

    with Session(engine) as session:
        memories = (
            session.query(Memoria)
            .filter(Memoria.metadados["user_id"].astext == user_id)
            .order_by(Memoria.embedding.max_inner_product(embedding))
            .limit(limit)
            .all()
        )

    return [
        SimilaritySearchResponse(
            id=str(memory.id),
            metadata=memory.metadados,
            page_content=memory.metadados["data"],
        )
        for memory in memories
    ]


@app.post("/add-embeddings")
def add_texts(payload: AddEmbeddingsPayload) -> list[str]:

    with Session(engine) as session:

        memories = [
            Memoria(
                id=id,
                embedding=embedding,
                metadados=metadadata,
            )
            for embedding, metadadata, id in zip(
                payload.embeddings, payload.metadatas, payload.ids
            )
        ]

        session.add_all(memories)
        session.commit()

    return payload.ids


@app.post("/add-texts")
def add_texts(payload: AddTextsPayload) -> list[str]:

    embeddings = embedder.embed_documents(payload.texts)

    with Session(engine) as session:

        memories = [
            Memoria(
                id=id,
                embedding=embedding,
                metadados=metadadata,
            )
            for embedding, metadadata, id in zip(
                embeddings, payload.metadatas, payload.ids
            )
        ]

        session.add_all(memories)
        session.commit()

    return payload.ids
