from typing import Any, Iterable, Optional, Sequence, TypeVar

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

VST = TypeVar("VST", bound="VectorStore")


class PGVectorRemote(VectorStore):
    def __init__(self, embedder: Embeddings) -> None:
        self.embedder = embedder

    def from_texts(
        cls: type[VST],
        texts: list[str],
        embedding: Embeddings,
        metadatas: None | list[dict] = None,
        *,
        ids: None | list[str] = None,
        **kwargs: Any,
    ) -> VST:
        pass

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        pass

    def delete(self, ids: None | list[str] = None, **kwargs: Any) -> Optional[bool]:

        if ids is None:
            return

        response = requests.delete(
            "http://localhost:8082/delete",
            json=ids,
        )

        return response.json()["deleted"]

    def get_by_ids(self, ids: Sequence[str]) -> list[Document]:

        response = requests.get(
            "http://localhost:8082/get-by-ids", params={"ids": ",".join(ids)}
        )

        return [
            Document(
                metadata=document["metadata"],
                page_content=document["page_content"],
                id=document["id"],
            )
            for document in response.json()
        ]

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:

        response = requests.post(
            "http://localhost:8082/similarity-search-by-vector",
            json={"embedding": embedding, "k": k, "kwargs": kwargs},
        )

        return [
            Document(
                metadata=document["metadata"],
                page_content=document["page_content"],
                id=document["id"],
            )
            for document in response.json()
        ]

    def add_embeddings(
        self,
        embeddings: list[list[float]],
        texts: None | list[dict] = None,
        metadatas: None | list[dict] = None,
        ids: None | list[str] = None,
        **kwargs: Any,
    ):

        if metadatas[0].get("type") == "user_identity":
            return []

        requests.post(
            "http://localhost:8082/add-embeddings",
            json={"embeddings": embeddings, "metadatas": metadatas, "ids": ids},
        )

        return [ids]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: None | list[dict] = None,
        *,
        ids: None | list[str] = None,
        **kwargs: Any,
    ) -> list[str]:

        if texts == [""]:
            return []

        requests.post(
            "http://localhost:8082/add-texts",
            json={"texts": texts, "metadatas": metadatas, "ids": ids},
        )

        return [ids]
