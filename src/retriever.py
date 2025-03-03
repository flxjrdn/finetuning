from typing import List

from embedding import Embedder
from vector_store import VectorStore


N_CHUNKS = 3


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def get_chunks(self, query: str) -> List[str]:
        query_embedding = self.embedder.embed(query)
        retrieved_texts = self.vector_store.query(
            query_embedding=query_embedding,
            n_chunks=N_CHUNKS,
        )
        return retrieved_texts


if __name__ == "__main__":
    query = "Wie wird das Guthaben der Betriebsrente+ angelegt?"
    retriever = Retriever(
        vector_store=VectorStore(),
        embedder=Embedder(),
    )
    chunks = retriever.get_chunks(query)
    print(chunks)
